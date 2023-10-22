# %% Initialize model
import numpy as np
import torch
import sys
import os
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../.."))
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
# %% Get parameters and reconstruct
from src.utils.torch_utils import make_psd

feasible_lambda = 10

P_params = policy_net_state_dict['P_params'].unsqueeze(0)
H_params = policy_net_state_dict['H_params']
zeros_n = torch.zeros((1, n_qp, 1), device=device)
ones_m = torch.ones((1, m_qp, 1), device=device)
I = torch.eye(1, device=device).unsqueeze(0)
zeros_1 = torch.zeros((1, 1), device=device)
Pinv = make_psd(P_params, min_eig=1e-2)
tilde_P_inv = torch.cat([
    torch.cat([Pinv, zeros_n], dim=2),
    torch.cat([zeros_n.transpose(1, 2), 1 / feasible_lambda * I], dim=2)
], dim=1)
H = H_params.view(m_qp, n_qp).unsqueeze(0)
tilde_H = torch.cat([
    torch.cat([H, ones_m], dim=2),
    torch.cat([zeros_n.transpose(1, 2), I], dim=2)
], dim=1)
P = torch.linalg.inv(tilde_P_inv).squeeze(0).cpu().numpy()
H = tilde_H.squeeze(0).cpu().numpy()
Wq_params = policy_net_state_dict['qb_affine_layer.weight'].unsqueeze(0)
Wq_tilde = torch.cat([
    Wq_params,
    torch.zeros((1, 1, Wq_params.shape[2]), device=device),
], dim=1)
Wq = Wq_tilde.squeeze(0).cpu().numpy()

# %% Get control invariant set
from src.envs.env_creators import sys_param, env_creators
from src.utils.sets import compute_MCI
from src.utils.geometry import find_supporting_hyperplanes
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
A_MCI, b_MCI = find_supporting_hyperplanes(MCI)
# %% Dump parameters
np.savez(
    "parameters.npz",
    A=A,
    B=B,
    P=P,
    H=H,
    Wq=Wq,
    A_MCI=A_MCI,
    b_MCI=b_MCI,
)

# %%
