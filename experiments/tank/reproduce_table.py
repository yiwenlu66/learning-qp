# %%
from glob import glob
import pandas as pd
import numpy as np
import torch

def read_csv(short_name):
    wildcard = f"{short_name}_2*"
    filename = sorted(glob(f"test_results/{wildcard}"))[-1]
    return pd.read_csv(filename, dtype={"constraint_violated": "bool"})

def read_mpc_iter_count(short_name):
    wildcard = f"{short_name}_mpc_iter_count_2*"
    filename = sorted(glob(f"test_results/{wildcard}"))[-1]
    return np.genfromtxt(filename)


def affine_layer_flops(input_size, output_size, has_bias, has_relu):
    flops = 2 * input_size * output_size
    if not has_bias:
        flops -= output_size
    if has_relu:
        flops += output_size
    return flops

def qp_flops(n_sys, n_qp, m_qp, qp_iter):
    get_q_flops = affine_layer_flops(2 * n_sys, n_qp, False, False)
    get_b_flops = affine_layer_flops(n_sys, m_qp, True, False)
    get_mu_flops = affine_layer_flops(n_sys, m_qp, False, False) + affine_layer_flops(m_qp, m_qp, False, False) + m_qp
    iter_flops = m_qp   # Adding primal-dual variables
    iter_flops += 2 * m_qp * (m_qp - 1)   # Matrix-vector multiplication
    iter_flops += 5 * m_qp   # Vector additions
    return get_q_flops + get_b_flops + get_mu_flops + qp_iter * iter_flops

def mpc_flops(n_sys, m_sys, N, iter_count_arr):
    n_qp = m_sys * N
    m_qp = 2 * (m_sys + n_sys) * N
    min_iter = np.min(iter_count_arr)
    max_iter = np.max(iter_count_arr)
    median_iter = np.median(iter_count_arr)
    min_flops = qp_flops(n_sys, n_qp, m_qp, min_iter)
    max_flops = qp_flops(n_sys, n_qp, m_qp, max_iter)
    median_flops = qp_flops(n_sys, n_qp, m_qp, median_iter)
    return min_flops, max_flops, median_flops

def mlp_flops(input_size, output_size, hidden_sizes):
    flops = 0
    prev_size = input_size
    for size in hidden_sizes:
        flops += affine_layer_flops(prev_size, size, True, True)
        prev_size = size
    flops += affine_layer_flops(prev_size, output_size, True, False)
    return flops

def count_parameters(exp_name):
    checkpoint_path = f"runs/tank_{exp_name}/nn/tank.pth"
    checkpoint = torch.load(checkpoint_path)
    total_params = 0
    for key, value in checkpoint['model'].items():
        if key.startswith("a2c_network.policy_net") or key.startswith("a2c_network.actor_mlp"):
            total_params += value.numel()
    return total_params

def get_row(short_name, method, n_sys=4, m_sys=2, n_qp=None, m_qp=None, qp_iter=10, N_mpc=None, mlp_last_size=None):
    """Output (short name, success rate, cost, penalized costs, FLOPs, learnable parameters)."""
    result_df = read_csv(short_name)
    total_episodes = len(result_df)
    penalty = 100000
    avg_cost = result_df['cumulative_cost'].sum() / result_df['episode_length'].sum()
    avg_cost_penalized = (result_df['cumulative_cost'].sum() + penalty * result_df["constraint_violated"].sum()) / result_df['episode_length'].sum()
    freq_violation = result_df["constraint_violated"].sum() / result_df['episode_length'].sum()
    success_rate = 1. - result_df["constraint_violated"].sum() / total_episodes

    # Count FLOPs
    if method == "qp":
        flops = qp_flops(n_sys, n_qp, m_qp, qp_iter)
    elif method == "mpc":
        iter_count_arr = read_mpc_iter_count(short_name)
        flops = mpc_flops(n_sys, m_sys, N_mpc, iter_count_arr)
    elif method == "mlp":
        flops = mlp_flops(2 * n_sys, m_sys, [i * mlp_last_size for i in [4, 2, 1]])

    # Count learnable parameters
    if method == "mpc":
        num_param = 0
    else:
        num_param = count_parameters(short_name)

    return short_name, success_rate, avg_cost, avg_cost_penalized, flops, num_param

# %%
rows = [
    get_row("reproduce_mpc_2_0", "mpc", N_mpc=2),
    get_row("reproduce_mpc_2_1", "mpc", N_mpc=2),
    get_row("reproduce_mpc_2_10", "mpc", N_mpc=2),
    get_row("reproduce_mpc_2_100", "mpc", N_mpc=2),
    get_row("reproduce_mpc_4_0", "mpc", N_mpc=4),
    get_row("reproduce_mpc_4_1", "mpc", N_mpc=4),
    get_row("reproduce_mpc_4_10", "mpc", N_mpc=4),
    get_row("reproduce_mpc_4_100", "mpc", N_mpc=4),
    get_row("reproduce_mpc_8_0", "mpc", N_mpc=8),
    get_row("reproduce_mpc_8_1", "mpc", N_mpc=8),
    get_row("reproduce_mpc_8_10", "mpc", N_mpc=8),
    get_row("reproduce_mpc_8_100", "mpc", N_mpc=8),
    get_row("reproduce_mpc_16_0", "mpc", N_mpc=16),
    get_row("reproduce_mpc_16_1", "mpc", N_mpc=16),
    get_row("reproduce_mpc_16_10", "mpc", N_mpc=16),
    get_row("reproduce_mpc_16_100", "mpc", N_mpc=16),
    get_row("reproduce_mlp_8", "mlp", mlp_last_size=8),
    get_row("reproduce_mlp_16", "mlp", mlp_last_size=16),
    get_row("reproduce_mlp_32", "mlp", mlp_last_size=32),
    get_row("reproduce_mlp_64", "mlp", mlp_last_size=32),
    get_row("reproduce_qp_4_24", "qp", n_qp=4, m_qp=24),
    get_row("reproduce_qp_8_48", "qp", n_qp=8, m_qp=48),
    get_row("reproduce_qp_16_96", "qp", n_qp=16, m_qp=96),
]

df_result = pd.DataFrame(rows, columns=["name", "success_rate", "avg_cost", "avg_cost_penalized", "flops", "num_param"])
df_result.to_csv("test_results/reproduce_table.csv", index=False)
print(df_result)
