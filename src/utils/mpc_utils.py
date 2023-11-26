import torch
from .torch_utils import make_psd, bmv, kron
import numpy as np
import cvxpy as cp
from scipy.linalg import kron as sp_kron
from numpy.linalg import matrix_power as np_matrix_power
from scipy.linalg import block_diag
import do_mpc
from ..envs.mpc_baseline_parameters import get_mpc_baseline_parameters
import time


def generate_random_problem(bs, n, m, device):
    P_params = -1 + 2 * torch.rand((bs, n * (n + 1) // 2), device=device)
    q = -1 + 2 * torch.rand((bs, n), device=device)
    H_params = -1 + 2 * torch.rand((bs, m * n), device=device)
    x0 = -1 + 2 * torch.rand((bs, n), device=device)
    P = make_psd(P_params)
    H = H_params.view(-1, m, n)
    b = bmv(H, x0)
    return q, b, P, H


def mpc2qp(n_mpc, m_mpc, N, A, B, Q, R, x_min, x_max, u_min, u_max, x0, x_ref, normalize=False, Qf=None):
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
    - Qf (torch.Tensor, optional): Terminal state cost matrix, shape (n_mpc, n_mpc).

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

    Q_kron = torch.kron(torch.eye(N, device=A.device), Q)
    if Qf is not None:
        # Adjust the last block of Q_kron to include Qf
        Q_kron[-n_mpc:, -n_mpc:] += Qf

    q = -2 * XU.t().unsqueeze(0) @ Q_kron.unsqueeze(0) @ (kron(torch.ones((bs, N, 1), device=device), x_ref.unsqueeze(-1)) - Ax0.unsqueeze(-1))   # (bs, N * m_MPC, 1)
    q = q.squeeze(-1)  # (bs, N * m_MPC) = (bs, n)
    P = 2 * XU.t() @ Q_kron @ XU + 2 * kron(torch.eye(N, device=device), R)  # (n, n)
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


def mpc2qp_np(n_mpc, m_mpc, N, A, B, Q, R, x_min, x_max, u_min, u_max, x0, x_ref, normalize=False, Qf=None):
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
    - Qf (np.ndarray, optional): Terminal state cost matrix, shape (n_mpc, n_mpc).

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
    Q_kron = sp_kron(np.eye(N), Q)
    if Qf is not None:
        Q_kron[-n_mpc:, -n_mpc:] += Qf
    q = -2 * XU.T @ Q_kron @ (sp_kron(np.ones((N, 1)), x_ref.reshape(-1, 1)) - Ax0.reshape(-1, 1))
    q = q.squeeze()
    P = 2 * XU.T @ Q_kron @ XU + 2 * sp_kron(np.eye(N), R)

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

def scenario_robust_mpc(mpc_baseline_parameters, r):
    """
    Scenario-based robust MPC with process noise handling and constraints.

    Inputs:
    - mpc_baseline_parameters: Dict containing A, B, Q, R, Qf, disturbance magnitude, state bounds, input bounds, etc.

    Output: Function mapping from x0 to u0.
    """

    # Extract parameters
    A = mpc_baseline_parameters['A']
    B = mpc_baseline_parameters['B']
    Q = mpc_baseline_parameters['Q']
    R = mpc_baseline_parameters['R']
    n = mpc_baseline_parameters['n_mpc']
    m = mpc_baseline_parameters['m_mpc']
    Qf = mpc_baseline_parameters.get("terminal_coef", 0.) * np.eye(n)
    A_scenarios = mpc_baseline_parameters.get("A_scenarios", [A])
    B_scenarios = mpc_baseline_parameters.get("B_scenarios", [B])
    w_scenarios = mpc_baseline_parameters.get("w_scenarios", [np.zeros((n, 1))])
    x_min = mpc_baseline_parameters['x_min']
    x_max = mpc_baseline_parameters['x_max']
    u_min = mpc_baseline_parameters['u_min']
    u_max = mpc_baseline_parameters['u_max']

    # Define the model
    model = do_mpc.model.Model('discrete')

    # States, inputs, and noise variables
    x = model.set_variable('_x', 'x', shape=(n, 1))
    u = model.set_variable('_u', 'u', shape=(m, 1))
    w = model.set_variable('_p', 'w', shape=(n, 1))  # Process noise

    # Uncertain parameters
    Theta_A = model.set_variable('_p', 'Theta_A', shape=A.shape)
    Theta_B = model.set_variable('_p', 'Theta_B', shape=B.shape)

    # System dynamics including process noise
    model.set_rhs('x', Theta_A @ x + Theta_B @ u + w)

    # Setup model
    model.setup()

    # MPC controller
    mpc = do_mpc.controller.MPC(model)

    # MPC parameters
    setup_mpc = {
        'n_horizon': mpc_baseline_parameters['N'],
        'n_robust': 1,   # Exponential growth, so only 1 is reasonable
        't_step': 0.1,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    # Uncertain parameter scenarios
    mpc.set_uncertainty_values(
        Theta_A=np.array(A_scenarios),
        Theta_B=np.array(B_scenarios),
        w=np.array(w_scenarios),
    )

    # Constraints on states and inputs
    eps = 1e-3
    mpc.bounds['lower','_x', 'x'] = x_min + eps
    mpc.bounds['upper','_x', 'x'] = x_max - eps
    mpc.bounds['lower','_u', 'u'] = u_min
    mpc.bounds['upper','_u', 'u'] = u_max

    # Objective function
    mterm = (x - r).T @ Qf @ (x - r)
    lterm = (x - r).T @ Q @ (x - r) + u.T @ R @ u
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Setup MPC
    mpc.setup()

    # Control function
    def mpc_control(x0, is_active=True):
        if is_active:
            t = time.time()
            mpc.x0 = x0

            # Solve the MPC problem
            u0 = mpc.make_step(x0)

            return u0.squeeze(-1), time.time() - t
        else:
            return np.zeros((m,)), 0.

    return mpc_control


def tube_robust_mpc(mpc_baseline_parameters, r):
    """
    Tube-based robust MPC with process noise handling and constraints.

    Inputs:
    - mpc_baseline_parameters: Dict containing A, B, Q, R, Qf, disturbance magnitude, state bounds, input bounds, etc.

    Output: Function mapping from x0 to u0.

    Reference: https://github.com/martindoff/DC-TMPC/; we only consider the case of LTI system (so that there is no successive linearization and no A2, B2).
    """
    # Extract parameters
    A = mpc_baseline_parameters['A']
    B = mpc_baseline_parameters['B']
    Q = mpc_baseline_parameters['Q']
    R = mpc_baseline_parameters['R']
    alpha = np.linalg.norm(Q)   # Here we assume Q, R are multiples of identity matrices
    beta = np.linalg.norm(R)
    n = mpc_baseline_parameters['n_mpc']
    m = mpc_baseline_parameters['m_mpc']
    Qf = mpc_baseline_parameters.get("terminal_coef", 0.) * np.eye(n)
    N = mpc_baseline_parameters['N']
    x_min = mpc_baseline_parameters['x_min']
    x_max = mpc_baseline_parameters['x_max']
    u_min = mpc_baseline_parameters['u_min']
    u_max = mpc_baseline_parameters['u_max']
    max_disturbance_per_dim = mpc_baseline_parameters.get('max_disturbance_per_dim', 0)

    # Compute feedback K
    def dp(A, B, Q, R, P):
        """ Implement one iteration of the DP recursion to compute K
        Input: state space matrices A and B, state penalty Q, input penalty R,
            Riccati equation solution P
        Output: gain K, P
        """

        # Compute gain
        S = np.linalg.inv(B.T @ P @ B + R)
        K = - S @ B.T @ P @ A

        # Update P
        P = Q + A.T @ P @ A - A.T @ P @ B @ S @ B.T @ P @ A

        return K, P

    P = Qf
    K = np.zeros((N + 1, m, n))          # gain matrix
    Phi = np.zeros((N + 1, n, n))       # closed-loop state transition matrix
    for l in reversed(range(N)):
        K[l, :, :], P = dp(A, B, Q, R, P)
        Phi[l, :, :] = A + B @ K[l, :, :]

    # Define optimization problem
    N_ver = 2 ** n                     # number of vertices

    # Optimization variables
    theta = cp.Variable(N + 1)               # cost
    v = cp.Variable((m, N))            # input perturbation
    s_low = cp.Variable((n, N + 1))    # state perturbation (lower bound)
    s_up = cp.Variable((n, N + 1))     # state perturbation (upper bound)
    s_ = {}                                # create dictionary for 3D variable
    ws = {}                            # Each item is a noise vector corresponding to a vertex
    for l in range(N_ver):
        s_[l] = cp.Expression
        ws[l] = np.zeros((n,))

    # Parameters (value set at run time)
    x0 = cp.Parameter(n)

    # Define blockdiag matrices for page-wise matrix multiplication
    K_ = block_diag(*K[:-1, :, :])
    Phi_ = block_diag(*Phi[:-1, :, :])
    B_ = block_diag(*([B] * N))

    # Objective
    objective = cp.Minimize(cp.sum(theta))

    # Constraints
    constr = []

    # Assemble vertices
    for l in range(N_ver):
        # Convert l to binary string
        l_bin = bin(l)[2:].zfill(n)
        # Map binary string to lows and ups
        mapping_str_to_ss = lambda c: s_low if c == '0' else s_up
        mapping_str_to_w = lambda c: -max_disturbance_per_dim if c == '0' else max_disturbance_per_dim
        ss = map(mapping_str_to_ss, l_bin)
        w = np.array(list(map(mapping_str_to_w, l_bin)))   # (n,) array
        s_[l] = cp.vstack([s[i, :] for (i, s) in enumerate(ss)])
        ws[l] = w

    for l in range(N_ver):
        # Define some useful variables
        s_r = cp.reshape(s_[l][:, :-1], (n * N, 1))
        v_r = cp.reshape(v, (m * N, 1))
        K_s = (K_ @ s_r).T
        Phi_s = cp.reshape(Phi_ @ s_r, ((n, N)))
        B_v = cp.reshape(B_ @ v_r, (n, N))
        K_s_r = cp.reshape(K_s, (m, N))

        # SOC objective constraints
        constr += [
            theta[:-1] >= \
                alpha * cp.square(s_[l][:, :-1] - np.expand_dims(r, -1)).sum(0) + \
                beta * cp.square(v + K_s_r).sum(0)
        ]

        constr += [
            theta[-1] >= cp.quad_form(s_[l][:, -1] - r, Qf)
        ]

        # Input constraints
        constr += [v + K_s_r >= u_min,
                   v + K_s_r <= u_max]

        # Tube
        constr += [
            s_low[:, 1:] <= Phi_s + B_v + np.expand_dims(ws[l], -1)
        ]

        constr += [
            s_up[:, 1:] >= \
                A @ s_[l][:, :-1] + B @ (v + K_s_r) + np.expand_dims(ws[l], -1)
        ]

    # State constraints
    constr += [
        s_low[:, :-1] >= x_min,
        s_up[:, :-1] >= x_min,
        s_up[:, :-1] <= x_max,
        s_low[:, :-1] <= x_max,
        s_low[:, 0] == x0,
        s_up[:, 0] == x0,
    ]

    # Define problem
    problem = cp.Problem(objective, constr)

    # Control function
    def mpc_control(x0_current, is_active=True):
        if is_active:
            t = time.time()
            x0.value = x0_current
            problem.solve(solver=cp.MOSEK, verbose=False, mosek_params={'MSK_IPAR_NUM_THREADS': 1})
            K_s = K[0, :, :] @ x0_current
            if v.value is not None:
                u0 = v.value[:, 0] + K_s
            else:
                # No solution, use default value
                u0 = np.zeros((m,))
            return u0, time.time() - t
        else:
            return np.zeros((m,)), 0.

    return mpc_control



if __name__ == "__main__":
    # Test scenario MPC
    mpc_baseline_parameters = get_mpc_baseline_parameters("double_integrator", 10)
    mpc_baseline_parameters["A_scenarios"] = [
        mpc_baseline_parameters["A"],
        1.1 * mpc_baseline_parameters["A"],
    ]
    mpc_baseline_parameters["B_scenarios"] = [
        mpc_baseline_parameters["B"],
        0.9 * mpc_baseline_parameters["B"],
        1.1 * mpc_baseline_parameters["B"],
    ]
    mpc_baseline_parameters["w_scenarios"] = [
        np.zeros((2, 1)),
        0.1 * np.ones((2, 1)),
        -0.1 * np.ones((2, 1)),
    ]
    controller = scenario_robust_mpc(mpc_baseline_parameters, np.zeros(2))

    # Test tube MPC
    mpc_baseline_parameters["max_disturbance_per_dim"] = 0.2
    controller = tube_robust_mpc(mpc_baseline_parameters, np.zeros(2))
