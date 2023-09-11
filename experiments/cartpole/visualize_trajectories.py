# %% Specify test case
import numpy as np

# Initial position and reference position
x0 = 0.
x_ref = 1.

# Controlling process noise and parametric uncertainty
noise_level = 0
parametric_uncertainty = True
parameter_randomization_seed = 42

# %% Set up test bench
import sys
import os
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../.."))

from envs.env_creators import sys_param, env_creators
from envs.mpc_baseline_parameters import get_mpc_baseline_parameters
from modules.qp_unrolled_network import QPUnrolledNetwork
import torch
from matplotlib import pyplot as plt


# Utilities

def obs_to_state(obs):
    # Convert obs in batch size 1 in form (x, x_ref, x_dot, sin(theta), cos(theta), theta_dot) to state (x, x_dot, theta, theta_dot)
    x, x_ref, x_dot, sin_theta, cos_theta, theta_dot = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], obs[:, 4], obs[:, 5]
    theta = torch.atan2(sin_theta, cos_theta)
    return torch.stack([x, x_dot, theta, theta_dot], dim=1).squeeze(0)

def make_obs(state, x_ref, running_mean, running_std, normalize):
    x, x_dot, theta, theta_dot = state
    raw_obs = torch.tensor(np.array([x, x_ref, x_dot, np.sin(theta), np.cos(theta), theta_dot]), device=device, dtype=torch.float)
    if not normalize:
        return raw_obs.unsqueeze(0)
    else:
        return ((raw_obs - running_mean) / running_std).unsqueeze(0)

def get_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["model"]
    prefix = "a2c_network.policy_net."
    policy_net_state_dict = {k.lstrip(prefix): v for (k, v) in model.items() if k.startswith(prefix)}
    running_mean = model["running_mean_std.running_mean"].to(dtype=torch.float)
    running_std = model["running_mean_std.running_var"].sqrt().to(dtype=torch.float)
    return policy_net_state_dict, running_mean, running_std

def rescale_action(action, low=-1., high=8.):
    action = action.clamp(-1., 1.)
    return low + (high - low) * (action + 1) / 2

t = lambda arr: torch.tensor(arr, device=device, dtype=torch.float).unsqueeze(0)
a = lambda t: t.detach().cpu().numpy()

# Constants and options
n_sys = 4
m_sys = 1
input_size = 6
n = 16
m = 32
qp_iter = 10
device = "cuda:0"


# # Learned QP
# net = QPUnrolledNetwork(device, input_size, n, m, qp_iter, None, True, True)
exp_name = f"shared_affine_noise{noise_level}_n{n}_m{m}"
# if parametric_uncertainty:
#     exp_name += "+rand"
# checkpoint_path = f"runs/tank_{exp_name}/nn/tank.pth"
# policy_net_state_dict, running_mean, running_std = get_state_dict(checkpoint_path)
# net.load_state_dict(policy_net_state_dict)
# running_mean, running_std = running_mean.to(device=device), running_std.to(device=device)
# net.to(device)

# MPC module
mpc_module = QPUnrolledNetwork(
    device, input_size, n, m, qp_iter, None, True, True,
    mpc_baseline=get_mpc_baseline_parameters("cartpole", 8),
    use_osqp_for_mpc=True,
)

# Environment
env = env_creators["cartpole"](
    noise_level=noise_level,
    bs=1,
    max_steps=300,
    keep_stats=True,
    run_name=exp_name,
    exp_name=exp_name,
    randomize=parametric_uncertainty,
)

# %% MLP Policy
import sys
mlp_exp_name = f"mlp_noise{noise_level}"
if parametric_uncertainty:
    mlp_exp_name += "+rand"
sys.argv = [""] + f"""test tank --num-parallel 1 \
        --noise-level {noise_level} \
        --exp-name {mlp_exp_name}""".split()
import run
mlp_checkpoint_path = f"runs/tank_{mlp_exp_name}/nn/tank.pth"
mlp_player = run.runner.create_player()
mlp_player.restore(mlp_checkpoint_path)

# %% Test for MPC
raw_obs = env.reset(t(x0), t(x_ref), randomize_seed=parameter_randomization_seed)
done = False


xs_mpc = [obs_to_state(raw_obs)]
us_mpc = []

while not done:
    u_all, problem_params = mpc_module(raw_obs, return_problem_params=True)
    u = u_all[:, :m_sys]
    raw_obs, reward, done_t, info = env.step(u)
    xs_mpc.append(obs_to_state(raw_obs))
    us_mpc.append(u[0, :])
    obs = raw_obs
    done = done_t.item()

# %% Test for learned QP
xs_qp = [t(x0).squeeze(0)]
us_qp = []
done = False
env.reset(t(x0), t(x_ref), randomize_seed=parameter_randomization_seed)
x = x0
obs = make_obs(x, x_ref, running_mean, running_std, True)
while not done:
    action_all, problem_params = net(obs, return_problem_params=True)
    u = rescale_action(action_all[:, :m_sys])
    raw_obs, reward, done_t, info = env.step(u)
    xs_qp.append(raw_obs[0, :4])
    us_qp.append(u[0, :])
    obs = (raw_obs - running_mean) / running_std
    done = done_t.item()

# %% Test for MLP
xs_mlp = [t(x0).squeeze(0)]
us_mlp = []
done = False
env.reset(t(x0), t(x_ref), randomize_seed=parameter_randomization_seed)
x = x0
obs = make_obs(x, x_ref, running_mean, running_std, False)
while not done:
    action = mlp_player.get_action(obs.squeeze(0), is_deterministic=True)
    obs, reward, done_t, info = env.step(action.unsqueeze(0))
    xs_mlp.append(obs[0, :4])
    us_mlp.append(action)
    done = done_t.item()

# %% Plot 1: cost curve
cost_mpc = [env.cost(x - t(x_ref), u.unsqueeze(0)).item() for (x, u) in zip(xs_mpc, us_mpc)]
cost_qp = [env.cost(x - t(x_ref), u.unsqueeze(0)).item() for (x, u) in zip(xs_qp, us_qp)]
cost_mlp = [env.cost(x - t(x_ref), u.unsqueeze(0)).item() for (x, u) in zip(xs_mlp, us_mlp)]

# Compute the baseline
baseline = min(min(cost_mpc), min(cost_qp), min(cost_mlp)) - 1e-2

# Deduct the baseline from each data series
cost_mpc_baseline = np.array(cost_mpc) - baseline
cost_qp_baseline = np.array(cost_qp) - baseline
cost_mlp_baseline = np.array(cost_mlp) - baseline

# Plotting
plt.title("Per-step LQ cost")
plt.plot(cost_mpc_baseline, label="MPC")
plt.plot(cost_qp_baseline, label="QP")
plt.plot(cost_mlp_baseline, label="MLP")

# Set y-axis to log scale
plt.yscale('log')

# Modify tick labels to show the true value
yticks = plt.yticks()[0]
plt.yticks(yticks, [f"{y + baseline:.0e}" for y in yticks])

plt.legend()

# %% Plot 2: Trajectory
# Create a 3-row, 2-column matrix of subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))

# Example to populate the subplots
for i in range(2):
    for j in range(2):
        ax = axes[i, j]
        subscript = 2 * i + j
        ax.plot([a(xs_mpc[k][subscript]) for k in range(len(xs_mpc))], label="MPC")
        # ax.plot([a(xs_qp[k][subscript]) for k in range(len(xs_qp))], label="QP")
        # ax.plot([a(xs_mlp[k][subscript]) for k in range(len(xs_mlp))], label="MLP")
        if subscript == 0:
            ax.axhline(y=x_ref, color='r', linestyle='--', label='Ref')
        ax.legend()
        ax.set_title(['x', 'x_dot', 'theta', 'theta_dot'][subscript])

i = 2
for j in range(1):
    ax = axes[i, j]
    ax.plot([a(us_mpc[k][j]) for k in range(len(us_mpc))], label="MPC")
    # ax.plot([a(us_qp[k][j]) for k in range(len(us_qp))], label="QP")
    # ax.plot([a(us_mlp[k][j]) for k in range(len(us_mlp))], label="MLP")
    ax.legend()
    ax.set_title(f'f')

plt.tight_layout()
plt.show()

# %%
