import torch
from torch.nn import functional as F
from contextlib import nullcontext, contextmanager
import numpy as np


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

def bsolve(A, B):
    """Compute solve(A, B) in batch mode, where the first dimension of A can be singleton."""
    if A.dim() == 3 and B.dim() == 2 and A.shape[0] == 1:
        return torch.linalg.solve(A.squeeze(0), B.t()).t()
    else:
        return torch.linalg.solve(A, B)

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


def interpolate_state_dicts(state_dict_1, state_dict_2, weight):
    return {
        key: (1 - weight) * state_dict_1[key] + weight * state_dict_2[key] for key in state_dict_1.keys()
    }


@contextmanager
def conditional_fork_rng(seed=None, condition=True):
    """
    Context manager for conditionally applying PyTorch's fork_rng.

    Parameters:
    - seed (int, optional): The seed value for the random number generator.
    - condition (bool): Determines whether to apply fork_rng or not.

    Yields:
    - None: Yields control back to the caller within the context.
    """
    if condition:
        with torch.random.fork_rng():
            if seed is not None:
                torch.manual_seed(seed)
            yield
    else:
        with nullcontext():
            yield

def get_rng(device, seed=None):
    """
    Get a random number generator.

    Parameters:
    - device (torch.device): The device to use for the random number generator.
    - seed (int, optional): The seed value for the random number generator.

    Returns:
    - torch.Generator: A random number generator.
    """
    return torch.Generator(device=device).manual_seed(seed) if seed is not None else torch.Generator(device=device)
