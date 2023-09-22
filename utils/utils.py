import torch
from torch.nn import functional as F
import numpy as np
import scipy
import qpsolvers
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext, contextmanager


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
    if A.dtype != torch.float:
        import ipdb; ipdb.set_trace()
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

def generate_random_problem(bs, n, m, device):
    P_params = -1 + 2 * torch.rand((bs, n * (n + 1) // 2), device=device)
    q = -1 + 2 * torch.rand((bs, n), device=device)
    H_params = -1 + 2 * torch.rand((bs, m * n), device=device)
    x0 = -1 + 2 * torch.rand((bs, n), device=device)
    P = make_psd(P_params)
    H = H_params.view(-1, m, n)
    b = bmv(H, x0)
    return q, b, P, H

def osqp_solve_qp_guarantee_return(
    P, q, G=None, h=None, A=None, b=None, lb=None, ub=None, initvals=None, verbose=False, **kwargs,
):
    problem = qpsolvers.problem.Problem(P, q, G, h, A, b, lb, ub)
    solution = qpsolvers.solvers.osqp_.osqp_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.x.dtype == np.float64 else np.zeros(q.shape[0])

def osqp_oracle(q, b, P, H):
    return osqp_solve_qp_guarantee_return(
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


def mpc2qp(n_mpc, m_mpc, N, A, B, Q, R, x_min, x_max, u_min, u_max, x0, x_ref, normalize=False):
    """
    Converts Model Predictive Control (MPC) problem parameters into Quadratic Programming (QP) form.

    Parameters:
    - n_mpc (int): Dimension of the state space.
    - m_mpc (int): Dimension of the input space.
    - N (int): Prediction horizon.
    - A (torch.Tensor): State transition matrix, shape (n_mpc, n_mpc).
    - B (torch.Tensor): Control input matrix, shape (n_mpc, m_mpc).
    - Q (torch.Tensor): State cost matrix, shape (n_mpc, n_mpc).
    - R (torch.Tensor): Control cost matrix, shape (m_mpc, m_mpc).
    - x_min (float): Lower state bounds.
    - x_max (float): Upper state bounds.
    - u_min (float): Lower control bounds.
    - u_max (float): Upper control bounds.
    - x0 (torch.Tensor): Initial state, shape (batch_size, n_mpc).
    - x_ref (torch.Tensor): Reference state, shape (batch_size, n_mpc).
    - normalize (bool): Whether to normalize the control actions. If set to True, the solution of the QP problem will be rescaled actions within range [-1, 1].

    Returns:
    - n (int): Number of decision variables.
    - m (int): Number of constraints.
    - P (torch.Tensor): QP cost matrix, shape (n, n).
    - q (torch.Tensor): QP cost vector, shape (batch_size, n).
    - H (torch.Tensor): Constraint matrix, shape (m, n).
    - b (torch.Tensor): Constraint bounds, shape (batch_size, m).

    The converted QP problem is in form:
        minimize    (1/2)x'Px + q'x
        subject to  Hx + b >= 0,

    Notes:
    - The function assumes that A, B, Q, R are single matrices, and x0 and x_ref are in batch.
    - All tensors are expected to be on the same device.
    """
    bs = x0.shape[0]
    device = x0.device

    Ax0 = torch.cat([bmv((torch.linalg.matrix_power(A, k + 1)).unsqueeze(0), x0) for k in range(N)], 1)   # (bs, N * n_mpc)
    m = 2 * (n_mpc + m_mpc) * N   # number of constraints
    n = m_mpc * N                 # number of decision variables

    b = torch.cat([
        Ax0 - x_min,
        x_max - Ax0,
        -u_min * torch.ones((bs, n), device=device),
        u_max * torch.ones((bs, n), device=device),
    ], 1)

    XU = torch.zeros((N, n_mpc, N, m_mpc), device=device)
    for k in range(N):
        for j in range(k + 1):
            XU[k, :, j, :] = (torch.linalg.matrix_power(A, k - j) @ B)
    XU = XU.flatten(0, 1).flatten(1, 2)   # (N * n_MPC, N * m_MPC)
    q = -2 * XU.t().unsqueeze(0) @ kron(torch.eye(N, device=device).unsqueeze(0), Q) @ (kron(torch.ones((bs, N, 1), device=device), x_ref.unsqueeze(-1)) - Ax0.unsqueeze(-1))   # (bs, N * m_MPC, 1)
    q = q.squeeze(-1)  # (bs, N * m_MPC) = (bs, n)
    P = 2 * XU.t() @ kron(torch.eye(N, device=device), Q) @ XU + 2 * kron(torch.eye(N, device=device), R)  # (n, n)
    H = torch.cat([XU, -XU, torch.eye(n, device=device), -torch.eye(n, device=device)], 0)  # (m, n)

    if normalize:
        # u = alpha * u_normalized + beta
        alpha = (u_max - u_min) / 2 * torch.ones((m_mpc,), device=device)   # (m_MPC,)
        beta = (u_max + u_min) / 2 * torch.ones((m_mpc,), device=device)    # (m_MPC,)
        Alpha = torch.diag_embed(alpha.repeat(N))  # (n, n)
        Beta = beta.repeat(N)  # (n,)
        P_nom = Alpha @ P @ Alpha    # (n,)
        q_nom = bmv(Alpha.unsqueeze(0), q + bmv(P, Beta).unsqueeze(0))    # (bs, n)
        H_nom = H @ Alpha    # (m, n)
        b_nom = (H @ Beta).unsqueeze(0) + b    # (bs, m)
        P, q, H, b = P_nom, q_nom, H_nom, b_nom

    return n, m, P, q, H, b

def _getindex(arr, i):
    if type(arr) == scipy.sparse.csc_matrix:
        return arr
    else:
        return arr[i] if arr.shape[0] > 1 else arr[0]

def _worker(i):
    f = _worker.f
    arrays = _worker.arrays
    return f(*[_getindex(arr, i) for arr in arrays])

def np_batch_op(f, *arrays):
    get_bs = lambda arr: 1 if type(arr) == scipy.sparse.csc_matrix else arr.shape[0]
    bs = max([get_bs(arr) for arr in arrays])
    _worker.f = f
    _worker.arrays = arrays
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(_worker, range(bs)))
        
    ret = np.concatenate([np.expand_dims(arr, 0) for arr in results], 0)
    return ret

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
