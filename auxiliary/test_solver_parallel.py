import numpy as np
import sys
import os
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, ".."))
from src.envs.mpc_baseline_parameters import get_mpc_baseline_parameters
from src.modules.qp_solver import QPSolver
from src.utils.mpc_utils import mpc2qp
from src.utils.osqp_utils import osqp_oracle
from src.utils.np_batch_op import np_batch_op
import torch
import time
import scipy
import functools
import csv


def compare(N, num_parallel, device="cuda:0", iterations=100, seed=42, max_cpu_workers=8):
    """
    Compare parallelized solver vs. OSQP on MPC problem with horizon N.
    """

    # Load model and config
    mpc_baseline_parameters = get_mpc_baseline_parameters("tank", N)
    n_mpc = mpc_baseline_parameters["n_mpc"]
    m_mpc = mpc_baseline_parameters["m_mpc"]
    Qf = np.eye(n_mpc)    # Set terminal cost of MPC as needed
    mpc_baseline_parameters["Qf"] = Qf
    x_min = mpc_baseline_parameters["x_min"]
    x_max = mpc_baseline_parameters["x_max"]
    u_min = mpc_baseline_parameters["u_min"]
    u_max = mpc_baseline_parameters["u_max"]
    A = mpc_baseline_parameters["A"]
    B = mpc_baseline_parameters["B"]
    Q = mpc_baseline_parameters["Q"]
    R = mpc_baseline_parameters["R"]

    # Generate current state
    t = lambda a: torch.tensor(a, device=device, dtype=torch.float)
    x = t(x_min).unsqueeze(0) + t(x_max - x_min).unsqueeze(0) * torch.rand((num_parallel, 2 * n_mpc), device=device)

    # Translate to QP problem
    eps = 1e-3
    n, m, P, q, H, b = mpc2qp(
        n_mpc,
        m_mpc,
        N,
        t(A),
        t(B),
        t(Q),
        t(R),
        x_min + eps,
        x_max - eps,
        u_min,
        u_max,
        *mpc_baseline_parameters["obs_to_state_and_ref"](x),
        normalize=mpc_baseline_parameters.get("normalize", False),
        Qf=t(Qf),
    )

    # Time solving with GPU parallelized solver
    solver = QPSolver(device, n, m, P=P, H=H)
    t = time.time()
    Xs, primal_sols = solver(q, b, iters=iterations)
    t_parallel = time.time() - t

    # Time solving with OSQP
    f = lambda t: t.detach().cpu().numpy()
    f_sparse = lambda t: scipy.sparse.csc_matrix(t.cpu().numpy())
    osqp_oracle_with_iter_count = functools.partial(osqp_oracle, return_iter_count=True, max_iter=iterations)
    q_np, b_np, P_np, H_np = f(q), f(b), f_sparse(P), f_sparse(H)
    t = time.time()
    sol_np, iter_counts = np_batch_op(osqp_oracle_with_iter_count, q_np, b_np, P_np, H_np, max_workers=max_cpu_workers)
    t_osqp = time.time() - t

    return n, m, t_parallel, t_osqp

Ns_mpc = [2 ** i for i in range(1, 5)]
nums_parallel = [2 ** i for i in range(1, 16)]
func_input = [(N, num_parallel) for N in Ns_mpc for num_parallel in nums_parallel]
func_output = [compare(*args) for args in func_input]
# Write to CSV
with open("parallel_vs_osqp.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["N_mpc", "num_parallel", "n_qp", "m_qp", "t_parallel", "t_osqp"])
    for args, output in zip(func_input, func_output):
        writer.writerow([*args, *output])
