from .env_creators import sys_param
import numpy as np
import torch

def get_mpc_baseline_parameters(env_name, N):
    mpc_parameters = {
        "n_mpc": sys_param[env_name]["n"],
        "m_mpc": sys_param[env_name]["m"],
        "N": N,
        **sys_param[env_name],
    }
    if env_name == "tank":
        # Compute state and ref from obs: the first n entries of obs is state, and the latter n entries are ref
        mpc_parameters["obs_to_state_and_ref"] = lambda obs: (obs[:, :mpc_parameters["n_mpc"]], obs[:, mpc_parameters["n_mpc"]:])
    if env_name == "cartpole":
        # Compute A, B matrices for linearized system
        m_pole = mpc_parameters["m_pole_nom"]
        m_cart = mpc_parameters["m_cart_nom"]
        l = mpc_parameters["l_nom"]
        g = 9.8

        # Continuous time A, B matrices
        A_ct = np.array([
            [0, 1, 0, 0],
            [0, 0, -g * m_pole / m_cart, 0],
            [0, 0, 0, 1],
            [0, 0, (m_cart + m_pole) * g / (l * m_cart) , 0],
        ])
        B_ct = np.array([
            [0],
            [1 / m_cart],
            [0],
            [-1 / (l * m_cart)],
        ])

        # Discretization
        dt = sys_param[env_name]["dt"]
        A = np.eye(4) + dt * A_ct
        B = dt * B_ct

        mpc_parameters["A"] = A
        mpc_parameters["B"] = B

        # Compute state and ref from obs: obs is in format (x, x_ref, x_dot, sin_theta, cos_theta, theta_dot)
        def obs_to_state_and_ref(obs):
            x, x_dot, theta, theta_dot, x_ref = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], obs[:, 4]
            state = torch.stack([x, x_dot, theta, theta_dot], dim=1)
            zeros = torch.zeros_like(x_ref)
            ref = torch.stack([x_ref, zeros, zeros, zeros], dim=1)
            return state, ref
        mpc_parameters["obs_to_state_and_ref"] = obs_to_state_and_ref

    return mpc_parameters
