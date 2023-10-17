import torch
from .torch_utils import make_psd, bmv, kron
import numpy as np
from scipy.linalg import kron as sp_kron
from numpy.linalg import matrix_power as np_matrix_power


def generate_random_problem(bs, n, m, device):
    P_params = -1 + 2 * torch.rand((bs, n * (n + 1) // 2), device=device)
    q = -1 + 2 * torch.rand((bs, n), device=device)
    H_params = -1 + 2 * torch.rand((bs, m * n), device=device)
    x0 = -1 + 2 * torch.rand((bs, n), device=device)
    P = make_psd(P_params)
    H = H_params.view(-1, m, n)
    b = bmv(H, x0)
    return q, b, P, H


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


def mpc2qp_np(n_mpc, m_mpc, N, A, B, Q, R, x_min, x_max, u_min, u_max, x0, x_ref, normalize=False):
    """
    Converts Model Predictive Control (MPC) problem parameters into Quadratic Programming (QP) form using NumPy.

    Parameters:
    - n_mpc (int): Dimension of the state space.
    - m_mpc (int): Dimension of the input space.
    - N (int): Prediction horizon.
    - A (np.ndarray): State transition matrix, shape (n_mpc, n_mpc).
    - B (np.ndarray): Control input matrix, shape (n_mpc, m_mpc).
    - Q (np.ndarray): State cost matrix, shape (n_mpc, n_mpc).
    - R (np.ndarray): Control cost matrix, shape (m_mpc, m_mpc).
    - x_min (float): Lower state bounds.
    - x_max (float): Upper state bounds.
    - u_min (float): Lower control bounds.
    - u_max (float): Upper control bounds.
    - x0 (np.ndarray): Initial state, shape (n_mpc,).
    - x_ref (np.ndarray): Reference state, shape (n_mpc,).
    - normalize (bool): Whether to normalize control actions.

    Returns:
    - n (int): Number of decision variables.
    - m (int): Number of constraints.
    - P (np.ndarray): QP cost matrix, shape (n, n).
    - q (np.ndarray): QP cost vector, shape (n,).
    - H (np.ndarray): Constraint matrix, shape (m, n).
    - b (np.ndarray): Constraint bounds, shape (m,).

    Notes:
    - The function assumes that A, B, Q, R, x0, and x_ref are NumPy arrays.
    """

    # Compute Ax0 based on state transition matrix and initial state
    Ax0 = np.hstack([np_matrix_power(A, k + 1) @ x0 for k in range(N)])

    # Number of constraints and decision variables
    m = 2 * (n_mpc + m_mpc) * N
    n = m_mpc * N

    # Construct constraint bounds vector
    b = np.hstack([
        Ax0 - x_min,
        x_max - Ax0,
        -u_min * np.ones(n),
        u_max * np.ones(n),
    ])

    # Compute input-state mapping matrix
    XU = np.zeros((N, n_mpc, N, m_mpc))
    for k in range(N):
        for j in range(k + 1):
            XU[k, :, j, :] = np_matrix_power(A, k - j) @ B
    XU = XU.reshape(N * n_mpc, N * m_mpc)

    # Compute QP cost vector and matrix
    q = -2 * XU.T @ sp_kron(np.eye(N), Q) @ (sp_kron(np.ones((N, 1)), x_ref.reshape(-1, 1)) - Ax0.reshape(-1, 1))
    q = q.squeeze()
    P = 2 * XU.T @ sp_kron(np.eye(N), Q) @ XU + 2 * sp_kron(np.eye(N), R)

    # Compute constraint matrix
    H = np.vstack([XU, -XU, np.eye(n), -np.eye(n)])

    if normalize:
        # Normalization parameters
        alpha = (u_max - u_min) / 2 * np.ones(m_mpc)
        beta = (u_max + u_min) / 2 * np.ones(m_mpc)
        Alpha = np.diag(alpha.repeat(N))
        Beta = beta.repeat(N)

        # Update QP parameters with normalized versions
        P = Alpha @ P @ Alpha
        q = Alpha @ (q + P @ Beta)
        H = H @ Alpha
        b = H @ Beta + b

    return n, m, P, q, H, b
