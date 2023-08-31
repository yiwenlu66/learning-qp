import pandas as pd
from glob import glob

df = pd.DataFrame(columns=[
    "Noise level",
    "Method",
    "Horizon",
    "Num of variables",
    "Num of constraints",
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

for noise_level in [0, 0.1, 0.2, 0.5]:
    mlp_df = read_csv(f"mlp_noise{noise_level}_*")
    df.loc[len(df)] = [
        noise_level,
        "MLP (trained with noise_level=0.5)",
        "-",
        "-",
        "-",
        *get_stat(mlp_df),
    ]
    for n in [2, 4, 8, 16]:
        for m in [2, 4, 8, 16]:
            qp_df = read_csv(f"N0_n{n}_m{m}_noise{noise_level}_*")
            df.loc[len(df)] = [
                noise_level,
                "QP",
                "-",
                n,
                m,
                *get_stat(qp_df),
            ]
    for N in [1, 2, 4, 8, 16]:
        mpc_df = read_csv(f"N{N}_noise{noise_level}_*")
        df.loc[len(df)] = [
            noise_level,
            "MPC",
            N,
            2 * N,
            4 * N,
            *get_stat(mpc_df),
        ]

df.to_csv("benchmark_stat.csv", index=False)
