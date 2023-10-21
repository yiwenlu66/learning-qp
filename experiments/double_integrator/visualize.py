# %% Load system and compute maximal invariant set
import numpy as np
import sys
import os
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../.."))

from src.envs.env_creators import sys_param, env_creators
from src.utils.sets import compute_MCI
from matplotlib import pyplot as plt

A = sys_param["double_integrator"]["A"]
B = sys_param["double_integrator"]["B"]
Q = sys_param["double_integrator"]["Q"]
R = sys_param["double_integrator"]["R"]
x_min_scalar = sys_param["double_integrator"]["x_min"]
x_max_scalar = sys_param["double_integrator"]["x_max"]
u_min_scalar = sys_param["double_integrator"]["u_min"]
u_max_scalar = sys_param["double_integrator"]["u_max"]
x_min = x_min_scalar * np.ones(2)
x_max = x_max_scalar * np.ones(2)
u_min = u_min_scalar * np.ones(1)
u_max = u_max_scalar * np.ones(1)

MCI = compute_MCI(A, B, x_min, x_max, u_min, u_max, iterations=100)

fig, ax = plt.subplots()
# ax.fill(X0_vertices[:, 0], X0_vertices[:, 1], alpha=0.3, label='Initial Set $X_0$', color='g')
ax.fill(MCI[:, 0], MCI[:, 1],
    alpha=0.7, label='Maximal Control Invariant Set', color='r')
ax.grid()

# %% Define MPC on the system
from src.utils.mpc_utils import mpc2qp_np
from src.utils.osqp_utils import osqp_oracle

N_mpc = 3    # The short horizon will make naive MPC fail, as shown in http://cse.lab.imtlucca.it/~bemporad/publications/papers/BBMbook.pdf, p. 247

def mpc_controller(x):
    """
    MPC controller for the double integrator system.
    """
    _, _, P, q, H, b = mpc2qp_np(
        n_mpc=2, m_mpc=1, N=N_mpc, A=A, B=B, Q=Q, R=R,
        x_min=x_min_scalar, x_max=x_max_scalar, u_min=u_min_scalar, u_max=u_max_scalar,
        x0=x, x_ref=np.zeros(2), normalize=False,
    )
    sol = osqp_oracle(q, b, P, H)
    return np.clip(sol[:1], u_min_scalar, u_max_scalar)

# %% Define learned controller on the system
from src.modules.qp_unrolled_network import QPUnrolledNetwork
import torch

def get_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["model"]
    prefix = "a2c_network.policy_net."
    policy_net_state_dict = {k.lstrip(prefix): v for (k, v) in model.items() if k.startswith(prefix)}
    if "running_mean_std.running_mean" in model:
        running_mean = model["running_mean_std.running_mean"].to(dtype=torch.float)
        running_std = model["running_mean_std.running_var"].sqrt().to(dtype=torch.float)
    else:
        running_mean = torch.tensor([0.])
        running_std = torch.tensor([1.])
    return policy_net_state_dict, running_mean, running_std

device = "cuda:0"
n_qp = 3
m_qp = 9
qp_iter = 10
symmetric = True
no_b = True
net = QPUnrolledNetwork(device, 2, n_qp, m_qp, qp_iter, None, True, True, force_feasible=True, symmetric=symmetric, no_b=no_b)
if not symmetric:
    exp_name = "default"
elif not no_b:
    exp_name = "symmetric"
else:
    exp_name = "symmetric_no_b"
checkpoint_path = f"runs/double_integrator_{exp_name}/nn/double_integrator.pth"
policy_net_state_dict, running_mean, running_std = get_state_dict(checkpoint_path)
net.load_state_dict(policy_net_state_dict)
running_mean, running_std = running_mean.to(device=device), running_std.to(device=device)
net.to(device)

t = lambda arr: torch.tensor(arr, device=device, dtype=torch.float).unsqueeze(0)
a = lambda t: t.squeeze(0).detach().cpu().numpy()

def learned_controller(x):
    sol = a(net(t(x)))
    sol *= 0.5    # Denormalize
    return np.clip(sol[:1], u_min_scalar, u_max_scalar)

# %% Define closed-loop dynamics

def get_cl_dynamics(controller):
    def g(x):
        return A @ x + B @ controller(x)
    return g

g_mpc = get_cl_dynamics(mpc_controller)
g_learned = get_cl_dynamics(learned_controller)


# %% Compute one-step reachable sets starting from MCI
from src.utils.sets import one_step_forward_reachable_set

reachable_mpc = one_step_forward_reachable_set(g_mpc, MCI, x_min, x_max)
reachable_learned = one_step_forward_reachable_set(g_learned, MCI, x_min, x_max)

fig, ax = plt.subplots()
ax.fill(MCI[:, 0], MCI[:, 1],
    alpha=0.3, label='Maximal Control Invariant Set', color='r')
ax.fill(reachable_mpc[:, 0], reachable_mpc[:, 1],
    alpha=0.3, label='One-step reachable set (MPC)', color='g')
ax.fill(reachable_learned[:, 0], reachable_learned[:, 1],
    alpha=0.3, label='One-step reachable set (Learned)', color='b')
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")

ax.legend()

# %% Compute positive invariant sets under closed-loop dynamics
from src.utils.sets import compute_positive_invariant_set_from_origin

pis_mpc = compute_positive_invariant_set_from_origin(g_mpc, x_min, x_max, initial_radius=1.5, iterations=150)
pis_learned = compute_positive_invariant_set_from_origin(g_learned, x_min, x_max, initial_radius=1.8, iterations=20)


# %%
fig, ax = plt.subplots()
ax.fill(MCI[:, 0], MCI[:, 1],
    alpha=1.0, label='Maximal Control Invariant Set', color='r')
ax.fill(pis_learned[:, 0], pis_learned[:, 1],
    alpha=1.0, label='Positive invariant set (Learned)', color='b')
ax.fill(pis_mpc[:, 0], pis_mpc[:, 1],
    alpha=1.0, label='Positive invariant set (MPC)', color='g')
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.grid()
ax.legend()


# %% Case study
from matplotlib.patches import Rectangle

def get_trajectory(controller, x0, max_steps=200):
    g = get_cl_dynamics(controller)
    x = x0
    xs = [x]
    total_cost = 0.
    for _ in range(max_steps):
        u = controller(x)
        total_cost += x.T @ Q @ x + u.T @ R @ u
        x = g(x)
        xs.append(x)
        if not (x_min <= x).all() or not (x <= x_max).all():
            break
    average_cost = total_cost / len(xs)
    return np.array(xs), average_cost


def plot_comparison(x0, mark='^'):
    traj_mpc, cost_mpc = get_trajectory(mpc_controller, x0)
    traj_learned, cost_learned = get_trajectory(learned_controller, x0)

    fig, ax = plt.subplots()
    ax.fill(MCI[:, 0], MCI[:, 1],
        alpha=0.1, label='Maximal Control Invariant Set', color='r')
    ax.fill(pis_mpc[:, 0], pis_mpc[:, 1],
        alpha=0.3, label='Positive invariant set (MPC)', color='g')
    ax.fill(pis_learned[:, 0], pis_learned[:, 1],
        alpha=0.3, label='Positive invariant set (Learned)', color='b')
    ax.plot(traj_mpc[:, 0], traj_mpc[:, 1], f'-{mark}', color='g', label=f"Trajectory (MPC) - Avg. Cost: {cost_mpc:.2f}")
    ax.plot(traj_learned[:, 0], traj_learned[:, 1], f'-{mark}', color='b', label="Trajectory (Learned) - Avg. Cost: {:.2f}".format(cost_learned))
    ax.grid()
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    # Plot the box constraint
    rect = Rectangle((x_min[0], x_min[1]), x_max[0] - x_min[0], x_max[1] - x_min[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_ylim(-3, 3)

    ax.legend()
    return fig, ax

# %%
fig, ax = plot_comparison(np.array([-4, 2.7]), '')
# ax.set_xlim(-1, 1)
# ax.set_ylim(-0.2, 0.2)

# %%
