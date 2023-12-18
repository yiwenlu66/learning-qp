import pandas as pd
import torch
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run_name", type=str, default="")

df = pd.DataFrame(columns=[
    "Noise level",
    "Parametric uncertainty",
    "Method",
    "Horizon",
    "Num of variables",
    "Num of constraints",
    "Num of learnable policy parameters",
    "Average cost",
    "Average cost (with penalty)",
    "Frequency of constraint violation (x1000)",
])

def read_csv(wildcard):
    filename = sorted(glob(f"test_results/{wildcard}"))[-1]
    return pd.read_csv(filename, dtype={"constraint_violated": "bool"})

def get_stat(df):
    max_episode_length = df['episode_length'].max()
    penalty = 100000
    avg_cost = df['cumulative_cost'].sum() / df['episode_length'].sum()
    avg_cost_penalized = (df['cumulative_cost'].sum() + penalty * df["constraint_violated"].sum()) / df['episode_length'].sum()
    freq_violation = df["constraint_violated"].sum() / df['episode_length'].sum()
    return avg_cost, avg_cost_penalized, freq_violation * 1000

def count_parameters(exp_name):
    checkpoint_path = f"runs/tank_{exp_name}/nn/tank.pth"
    checkpoint = torch.load(checkpoint_path)
    total_params = 0
    for key, value in checkpoint['model'].items():
        if key.startswith("a2c_network.policy_net") or key.startswith("a2c_network.actor_mlp"):
            total_params += value.numel()
    return total_params

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.run_name:
        # Iterate over all configurations
        for noise_level in [0, 0.1, 0.2, 0.5]:
            for rand in [False, True]:
                try:
                    wildcard = f"mlp_noise{noise_level}{'_rand' if rand else ''}_2*"
                    mlp_df = read_csv(wildcard)
                    df.loc[len(df)] = [
                        noise_level,
                        rand,
                        "MLP",
                        "-",
                        "-",
                        "-",
                        count_parameters(f"mlp_noise{noise_level}"),
                        *get_stat(mlp_df),
                    ]
                except:
                    print(f"Error reading file: {wildcard}")

                for n in [2, 4, 8, 16]:
                    for m in [2, 4, 8, 16, 32, 64]:
                        try:
                            wildcard = f"N0_n{n}_m{m}_noise{noise_level}{'_rand' if rand else ''}_2*"
                            qp_df = read_csv(wildcard)
                            df.loc[len(df)] = [
                                noise_level,
                                rand,
                                "QP",
                                "-",
                                n,
                                m,
                                count_parameters(f"shared_affine_noise{noise_level}_n{n}_m{m}"),
                                *get_stat(qp_df),
                            ]
                        except:
                            print(f"Error reading file: {wildcard}")

                for N in [1, 2, 4, 8, 16]:
                    try:
                        wildcard = f"N{N}_noise{noise_level}{'_rand' if rand else ''}_2*"
                        mpc_df = read_csv(wildcard)
                        df.loc[len(df)] = [
                            noise_level,
                            rand,
                            "MPC",
                            N,
                            2 * N,
                            12 * N,
                            0,
                            *get_stat(mpc_df),
                        ]
                    except:
                        print(f"Error reading file: {wildcard}")
        df.to_csv("benchmark_stat.csv", index=False)
    else:
        # Stat for particular run
        run_name = args.run_name
        wildcard = f"{run_name}_2*"
        raw_df = read_csv(wildcard)
        avg_cost, avg_cost_penalized, freq_violation = get_stat(raw_df)
        df.loc[len(df)] = [
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            avg_cost,
            avg_cost_penalized,
            freq_violation,
        ]
        df.to_csv(f"benchmark_stat_{run_name}.csv", index=False)
