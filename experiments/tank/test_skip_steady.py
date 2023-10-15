# %% Problem setup
import sys
import os
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../.."))

import numpy as np
from src.envs.env_creators import sys_param, env_creators
x_ref = np.array([19., 19., 2., 2.])
A = sys_param["tank"]["A"]
B = sys_param["tank"]["B"]
Q = sys_param["tank"]["Q"]
R = sys_param["tank"]["R"]
x_min = sys_param["tank"]["x_min"] * np.ones(4)
x_max = sys_param["tank"]["x_max"] * np.ones(4)
u_min = sys_param["tank"]["u_min"] * np.ones(2)
u_max = 1.0 * np.ones(2)

# %% Oracle
from src.utils.osqp_utils import osqp_oracle

# min (x - x_ref)' * Q * (x - x_ref) + u' * R * u, s.t., x = (I - A)^{-1} * B * u, x_min <= x <= x_max, u_min <= u <= u_max; cast into min 0.5 * u' * P * u + q' * u, s.t., H * u + b >= 0

inv_I_minus_A = np.linalg.inv(np.eye(A.shape[0]) - A)
P = 2 * (B.T @ inv_I_minus_A.T @ Q @ inv_I_minus_A @ B + R)

# Calculate q
q = -2 * inv_I_minus_A.T @ Q.T @ x_ref @ B

# Calculate c
c = x_ref.T @ Q @ x_ref

# Calculate H and b
H = np.vstack([
    inv_I_minus_A @ B,
    -inv_I_minus_A @ B,
    np.eye(u_min.shape[0]),
    -np.eye(u_max.shape[0])
])

b = np.hstack([
    -x_min,
    x_max,
    -u_min,
    u_max
])

u_opt = osqp_oracle(q, b, P, H)
x_opt = inv_I_minus_A @ B @ u_opt

# %% Evaluation
from icecream import ic
eval_value = lambda u: 0.5 * u.T @ P @ u + q.T @ u + c
opt_val = eval_value(u_opt)
ic(opt_val)

# %% Evaluate the learned controller
import torch
from src.modules.qp_unrolled_network import QPUnrolledNetwork

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

def make_obs(x, x_ref, running_mean, running_std, normalize):
    raw_obs = torch.tensor(np.concatenate([x, x_ref]), device=device, dtype=torch.float)
    if not normalize:
        return raw_obs.unsqueeze(0)
    else:
        return ((raw_obs - running_mean) / running_std).unsqueeze(0)


n_sys = 4
m_sys = 2
input_size = 8   # 4 for x, 4 for x_ref
n = 8
m = 32
qp_iter = 10
device = "cuda:0"

# Learned QP
net = QPUnrolledNetwork(device, input_size, n, m, qp_iter, None, True, True)
exp_name = "test_skip_steady"
checkpoint_path = f"runs/tank_{exp_name}/nn/tank.pth"
policy_net_state_dict, running_mean, running_std = get_state_dict(checkpoint_path)
net.load_state_dict(policy_net_state_dict)
running_mean, running_std = running_mean.to(device=device), running_std.to(device=device)
net.to(device)
obs = make_obs(0 * np.ones(4), x_ref, running_mean, running_std, False)
action_all, problem_params = net(obs, return_problem_params=True)
u = action_all[:, :2].squeeze(0).detach().cpu().numpy()
learned_val = eval_value(u)
ic(learned_val)


# %%
