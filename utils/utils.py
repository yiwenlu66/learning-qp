import torch
from torch.nn import functional as F
import numpy as np
import qpsolvers

def bmv(A, b):
    """Compute matrix multiply vector in batch mode."""
    bs = b.shape[0]
    if A.shape[0] == 1:
        # The same A for different b's; use matrix multiplication instead of broadcasting
        return (A.squeeze(0) @ b.t()).t()
    else:
        return (A @ b.unsqueeze(-1)).squeeze(-1)

def bma(A, B):
    """Batch-matrix-times-any, where any can be matrix or vector."""
    return (A @ B) if A.dim() == B.dim() else bmv(A, B)

def bvv(x, y):
    """Compute vector dot product in batch mode."""
    return bmv(x.unsqueeze(-2), y)

def bqf(x, A):
    """Compute quadratic form x' * A * x in batch mode."""
    return torch.einsum('bi,bij,bj->b', x, A, x)

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

def vectorize_upper_triangular(matrices):
    # Get the shape of the matrices
    b, n, _ = matrices.shape

    # Create the indices for the upper triangular part
    row_indices, col_indices = torch.triu_indices(n, n, device=matrices.device)

    # Create a mask of shape (b, n, n)
    mask = torch.zeros((b, n, n), device=matrices.device, dtype=torch.bool)
    
    # Set the upper triangular part of the mask to True
    mask[:, row_indices, col_indices] = True

    # Use the mask to extract the upper triangular part
    upper_triangular = matrices[mask]

    # Reshape the result to the desired shape
    upper_triangular = upper_triangular.view(b, -1)

    return upper_triangular

def generate_random_problem(bs, n, m, device):
    P_params = -1 + 2 * torch.rand((bs, n * (n + 1) // 2), device=device)
    q = -1 + 2 * torch.rand((bs, n), device=device)
    H_params = -1 + 2 * torch.rand((bs, m * n), device=device)
    x0 = -1 + 2 * torch.rand((bs, n), device=device)
    P = make_psd(P_params)
    H = H_params.view(-1, m, n)
    b = bmv(H, x0)
    return q, b, P, H

def osqp_oracle(q, b, P, H):
    return qpsolvers.solvers.osqp_.osqp_solve_qp(
        P=P, q=q, G=-H, h=b,
        A=None, b=None, lb=None, ub=None,
        max_iter=30000, eps_abs=1e-10, eps_rel=1e-10,eps_prim_inf=1e-10, eps_dual_inf=1e-10, verbose=False
    )

def interpolate_state_dicts(state_dict_1, state_dict_2, weight):
    return {
        key: (1 - weight) * state_dict_1[key] + weight * state_dict_2[key] for key in state_dict_1.keys()
    }

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


def mpc2qp(n_mpc, m_mpc, N, A, B, Q, R, x_min, x_max, u_min, u_max, x0, x_ref):
    bs = x0.shape[0]
    device = x0.device

    from icecream import ic; ic(x0, x_ref)
    Ax0 = torch.cat([bmv((torch.linalg.matrix_power(A, k + 1)).unsqueeze(0), x0) for k in range(N)], 1)   # (bs, N * n_mpc)
    m = 2 * (n_mpc + m_mpc) * N   # number of constraints
    n = m_mpc * N                 # number of decision variables

    b = torch.cat([
        Ax0 - x_min,
        x_max - Ax0,
        -u_min * torch.ones((bs, n), device=device),
        u_max * torch.ones((bs, n), device=device),
    ], 1)

    XU = torch.zeros((bs, N, n_mpc, N, m_mpc), device=device)
    for k in range(N):
        for j in range(k + 1):
            XU[:, k, :, j, :] = (torch.linalg.matrix_power(A, k - j) @ B).unsqueeze(0)
    XU = XU.flatten(1, 2).flatten(2, 3)   # (bs, N * n_MPC, N * m_MPC)
    q = -2 * XU.transpose(1, 2) @ kron(torch.eye(N, device=device).unsqueeze(0), Q) @ (kron(torch.ones((bs, N, 1), device=device), x_ref.unsqueeze(-1)) - Ax0.unsqueeze(-1))   # (bs, N * m_MPC, 1)
    q = q.squeeze(-1)  # (bs, N * m_MPC) = (bs, n)
    P = 2 * XU.transpose(-1, -2) @ kron(torch.eye(N, device=device).unsqueeze(0), Q.unsqueeze(0)) @ XU + 2 * kron(torch.eye(N, device=device).unsqueeze(0), R.unsqueeze(0))
    H = torch.cat([XU, -XU, torch.eye(n, device=device).unsqueeze(0).broadcast_to((bs, -1, -1)), torch.eye(n, device=device).unsqueeze(0).broadcast_to((bs, -1, -1))], 1)

    return n, m, P, q, H, b
