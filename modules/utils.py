import torch
from torch.nn import functional as F
import numpy as np

def make_psd(x, min_eig=0.1):
    """Assume x is (bs, N*(N+1)/2), create (bs, N, N) batch of PSD matrices using Cholesky."""
    bs, n_elem = x.shape
    N = (int(np.sqrt(1 + 8 * n_elem)) - 1) // 2
    cholesky_diag_index = torch.arange(N, dtype=torch.long) + 1
    cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1 # computes the indices of the future diagonal elements of the matrix
    elem = x.clone()
    elem[:, cholesky_diag_index] = np.sqrt(min_eig) + F.softplus(elem[:, cholesky_diag_index])
    tril_indices = torch.tril_indices(row=N, col=N, offset=0) # Collection that contains the indices of the non-zero elements of a lower triangular matrix
    cholesky = torch.zeros(size=(bs, N, N), dtype=torch.float, device=elem.device) #initialize a square matrix to zeros
    cholesky[:, tril_indices[0], tril_indices[1]] = elem # Assigns the elements of the vector to their correct position in the lower triangular matrix
    return cholesky @ cholesky.transpose(1, 2)

def bmv(A, b):
    """Compute matrix multiply vector in batch mode."""
    return (A @ b.unsqueeze(-1)).squeeze(-1)

def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)

def random_prob(device, bs, n_MPC, m_MPC, N, Ad, Bd, oracle_solver,
        x_bnd=[0, 20],
        u_bnd=[0, 8],
        x_sample_bnd=[0.5, 20],
        fixed_x_ref=None,
        ):
    Adt = torch.tensor(Ad, dtype=torch.float, device=device)
    Bdt = torch.tensor(Bd, dtype=torch.float, device=device)
    x_min, x_max = x_bnd
    u_min, u_max = u_bnd
    x_sample_min, x_sample_max = x_bnd
    x0 = x_sample_min + (x_sample_max - x_sample_min) * torch.rand((bs, n_MPC), device=device)
    if fixed_x_ref is None:
        x_ref = x_sample_min + (x_sample_max - x_sample_min) * torch.rand((bs, n_MPC), device=device)
    else:
        x_ref = torch.tensor(fixed_x_ref, device=device).unsqueeze(0).repeat((bs, 1))
    Ax0 = torch.zeros((bs, N, n_MPC), device=device)
    for k in range(N):
        Ax0[:, k, :] = bmv(torch.linalg.matrix_power(Adt, k + 1).unsqueeze(0), x0)
    Ax0 = Ax0.view((bs, N * n_MPC))
    n = m_MPC * N
    b = torch.cat([
        Ax0 - x_min,
        x_max - Ax0,
        -u_min * torch.ones((bs, n), device=device),
        u_max * torch.ones((bs, n), device=device),
    ], 1)
    XU = torch.zeros((bs, N, n_MPC, N, m_MPC), device=device)
    for k in range(N):
        for j in range(k + 1):
            XU[:, k, :, j, :] = (torch.linalg.matrix_power(Adt, k - j) @ Bdt).unsqueeze(0)
    XU = XU.flatten(1, 2).flatten(2, 3)   # (bs, N * n_MPC, N * m_MPC)
    P_MPC = torch.eye(n_MPC, device=device).unsqueeze(0)
    q = -2 * XU.transpose(1, 2) @ kron(torch.eye(N, device=device).unsqueeze(0), P_MPC) @ (kron(torch.ones((bs, N, 1), device=device), x_ref.unsqueeze(-1)) - Ax0.unsqueeze(-1))   # (bs, N * m_MPC, 1)
    q = q.squeeze(-1)  # (bs, N * m_MPC) = (bs, n)
    with torch.no_grad():
        Xs, sols = oracle_solver(q, b, iters=2000)   # (bs, iter + 1, 2 * m), (bs, iter + 1, n)
    oracle_X, oracle_sol = Xs[:, -1, :], sols[:, -1, :]   # (bs, n)
    return q, b, oracle_X, oracle_sol

def interpolate_state_dicts(state_dict_1, state_dict_2, weight):
    return {
        key: (1 - weight) * state_dict_1[key] + weight * state_dict_2[key] for key in state_dict_1.keys()
    }
