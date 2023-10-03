# %%
import os
import sys
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, ".."))
from modules.qp_solver import QPSolver
from modules.warm_starter import WarmStarter
from utils.torch_utils import bqf, bmv, bvv
from utils.mpc_utils import generate_random_problem
import torch
from torch.nn import functional as F
import numpy as np

n = 10
m = 5

device = "cuda:0"
torch.manual_seed(42)
q0, b0, P0, H0 = generate_random_problem(1, n, m, device)
q0_np = q0.squeeze(0).cpu().numpy()
b0_np = b0.squeeze(0).cpu().numpy()
P0_np = P0.squeeze(0).cpu().numpy()
H0_np = H0.squeeze(0).cpu().numpy()
np.save("example_problem.npy", {
    "q": q0_np,
    "b": b0_np,
    "P": P0_np,
    "H": H0_np,
})

solver = QPSolver(device, n, m, P=P0_np, H=H0_np)
ws = WarmStarter(device, n, m, fixed_P=True, fixed_H=True)
ws.load_state_dict(torch.load(f"models/warmstarter-{n}-{m}.pth"))
solver_ws = QPSolver(device, n, m, P=P0_np, H=H0_np,  warm_starter=ws)

iters = 1000
X, sol = solver(q0, b0, iters=iters)
X_ws, sol_ws = solver_ws(q0, b0, iters=iters)


# %%
from matplotlib import pyplot as plt
obj = [(0.5 * bqf(sol[:, i, :], P0) + bvv(sol[:, i, :], q0)).item() for i in range(sol.shape[1])]
obj_ws = [(0.5 * bqf(sol_ws[:, i, :], P0) + bvv(sol[:, i, :], q0)).item() for i in range(sol_ws.shape[1])]
plt.plot(obj)
plt.plot(obj_ws)

# %%
X_diff = [(X[:, i, :] - X[:, -1, :]).norm().item() for i in range(X.shape[1])]
X_diff_ws = [(X_ws[:, i, :] - X[:, -1, :]).norm().item() for i in range(X.shape[1])]
plt.plot(X_diff)
plt.plot(X_diff_ws)

# %%
