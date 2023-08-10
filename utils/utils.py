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
