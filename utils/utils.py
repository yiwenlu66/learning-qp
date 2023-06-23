import torch
from torch.nn import functional as F
import numpy as np

def bmv(A, b):
    """Compute matrix multiply vector in batch mode."""
    return (A @ b.unsqueeze(-1)).squeeze(-1)

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
    row_indices, col_indices = torch.triu_indices(n, n)

    # Expand dims and repeat for batch size
    row_indices = row_indices[None, :].expand(b, -1)
    col_indices = col_indices[None, :].expand(b, -1)

    # Use gather to extract the upper triangular part
    upper_triangular = matrices.gather(1, row_indices.unsqueeze(2)).gather(2, col_indices.unsqueeze(2))

    # Reshape the result to the desired shape
    upper_triangular = upper_triangular.view(b, n * (n + 1) // 2)

    return upper_triangular
