def bmv(A, b):
    """Compute matrix multiply vector in batch mode."""
    return (A @ b.unsqueeze(-1)).squeeze(-1)
