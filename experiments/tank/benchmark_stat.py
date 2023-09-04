import pandas as pd
import torch
from glob import glob

df = pd.DataFrame(columns=[
    "Noise level",
    "Method",
    "Horizon",
    "Num of variables",
    "Num of constraints",
    "Num of learnable policy parameters",
    "Average cost",
    "Frequency of constraint violation (x1000)",
])

def read_csv(wildcard):
    filename = sorted(glob(f"test_results/{wildcard}"))[-1]
    return pd.read_csv(filename, dtype={"constraint_violated": "bool"})

def get_stat(df):
    avg_cost = df['cumulative_cost'].sum() / df['episode_length'].sum()
    freq_violation = df["constraint_violated"].sum() / df['episode_length'].sum()
    return avg_cost, freq_violation * 1000

def count_parameters(exp_name):
    checkpoint_path = f"runs/tank_{exp_name}/nn/tank.pth"
    checkpoint = torch.load(checkpoint_path)
    total_params = 0
    for key, value in checkpoint['model'].items():
        if key.startswith("a2c_network.policy_net") or key.startswith("a2c_network.actor_mlp"):
            total_params += value.numel()
    return total_params

for noise_level in [0, 0.1, 0.2, 0.5]:
    mlp_df = read_csv(f"mlp_noise{noise_level}_*")
    df.loc[len(df)] = [
        noise_level,
        "MLP",
        "-",
        "-",
        "-",
        count_parameters(f"mlp_noise{noise_level}"),
        *get_stat(mlp_df),
    ]
    for n in [2, 4, 8, 16]:
        for m in [2, 4, 8, 16, 32, 64]:
            try:
                qp_df = read_csv(f"N0_n{n}_m{m}_noise{noise_level}_*")
                df.loc[len(df)] = [
                    noise_level,
                    "QP",
                    "-",
                    n,
                    m,
                    count_parameters(f"shared_affine_noise{noise_level}_n{n}_m{m}"),
                    *get_stat(qp_df),
                ]
            except:
                print(f"Error reading file: N0_n{n}_m{m}_noise{noise_level}_*")
    for N in [1, 2, 4, 8, 16]:
        mpc_df = read_csv(f"N{N}_noise{noise_level}_*")
        df.loc[len(df)] = [
            noise_level,
            "MPC",
            N,
            2 * N,
            12 * N,
            0,
            *get_stat(mpc_df),
        ]

df.to_csv("benchmark_stat.csv", index=False)
