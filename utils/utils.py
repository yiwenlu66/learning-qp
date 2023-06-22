import torch

def bmv(A, b):
    """Compute matrix multiply vector in batch mode."""
    return (A @ b.unsqueeze(-1)).squeeze(-1)

def bvv(x, y):
    """Compute vector dot product in batch mode."""
    return bmv(x.unsqueeze(-2), y)

def bqf(x, A):
    """Compute quadratic form x' * A * x in batch mode."""
    return torch.einsum('bi,bij,bj->b', x, A, x)
