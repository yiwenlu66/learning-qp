# %% Read test data from learned QP with MPC
from benchmark_stat import read_csv, get_stat
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

noise_level = 0.1
randomize = True
N = 4
n = 8
m = 32

randomize_flag = "_rand" if randomize else ""
df_mpc = read_csv(f"N{N}_noise{noise_level}{randomize_flag}_20*")
df_qp = read_csv(f"N0_n{n}_m{m}_noise{noise_level}{randomize_flag}_20*")
df_mlp = read_csv(f"mlp_noise{noise_level}{randomize_flag}_20*")

# %% Matrix of constraint violation

def violation_matrix(df1, df2):
    """
    Return four lists of indices standing for the trajectory indices where:
    1. constraint_violated=True in both df1 and df2
    2. constraint_violated=True in df1 but not df2
    3. constraint_violated=True in df2 but not df1
    4. constraint_violated=False in both df1 and df2
    """
    idx_both = []
    idx_df1 = []
    idx_df2 = []
    idx_none = []
    for i in range(len(df1)):
        if df1.iloc[i]["constraint_violated"] and df2.iloc[i]["constraint_violated"]:
            idx_both.append(i)
        elif df1.iloc[i]["constraint_violated"] and not df2.iloc[i]["constraint_violated"]:
            idx_df1.append(i)
        elif not df1.iloc[i]["constraint_violated"] and df2.iloc[i]["constraint_violated"]:
            idx_df2.append(i)
        else:
            idx_none.append(i)
    return idx_both, idx_df1, idx_df2, idx_none

violated_both, violated_mpc, violated_qp, violated_none = violation_matrix(df_mpc, df_qp)

data = {"MPC Success": [len(violated_none), len(violated_qp)], "MPC Fail": [len(violated_mpc), len(violated_both)]}
index_labels = ["QP Success", "QP Fail"]
df = pd.DataFrame(data=data, index=index_labels)
df

# %% Cost ratio histogram in the cases where both methods succeed
cost_mpc = df_mpc.iloc[violated_none]["cumulative_cost"]
cost_qp = df_qp.iloc[violated_none]["cumulative_cost"]
ratio = cost_mpc / cost_qp

n, bins, patches = plt.hist(ratio, bins=30, edgecolor='black', alpha=0.7)
max_freq = max(n)

# Add vertical dashed line at x=1
plt.axvline(x=1, color='r', linestyle='--')

# Annotations
text_y_pos = max_freq * 1.3
y_max = max_freq * 1.6
plt.arrow(0.8, text_y_pos, -0.6, 0, head_width=20, head_length=0.05, fc='black', ec='black')
plt.text(0.5, text_y_pos, 'MPC better', horizontalalignment='center', verticalalignment='bottom', color='black')
plt.arrow(1.2, text_y_pos, 0.6, 0, head_width=20, head_length=0.05, fc='black', ec='black')
plt.text(1.5, text_y_pos, 'Learned QP better', horizontalalignment='center', verticalalignment='bottom', color='black')

plt.xlabel('Ratio of average cost (MPC / Learned QP)')
plt.xlim(0, 2)
plt.ylim(0, y_max)


# %% Penalized cost ratio histogram in all cases
penalty = 100000
get_penalized_cost = lambda df: (df['cumulative_cost'] + penalty * df["constraint_violated"]) / df['episode_length']
penalized_cost_mpc = get_penalized_cost(df_mpc)
penalized_cost_qp = get_penalized_cost(df_qp)
penalized_cost_mlp = get_penalized_cost(df_mlp)

# Export penalized costs to csv; each row is (penalized_cost_mpc, penalized_cost_qp, penalized_cost_mlp)
header_line = "penalized_cost_mpc,penalized_cost_qp,penalized_cost_mlp"
np.savetxt("penalized_costs.csv", np.column_stack((penalized_cost_mpc, penalized_cost_qp, penalized_cost_mlp)), delimiter=",", header=header_line, comments='')



log_penalized_ratio = np.log10(penalized_cost_mpc / penalized_cost_qp)

n, bins, patches = plt.hist(log_penalized_ratio, bins=30, edgecolor='black', alpha=0.7)
max_freq = max(n)

# Add vertical dashed line at x=1
plt.axvline(x=0, color='r', linestyle='--')

# Annotations
text_y_pos = max_freq * 1.3
y_max = max_freq * 1.6
plt.arrow(-1, text_y_pos, -2, 0, head_width=50, head_length=0.05, fc='black', ec='black')
plt.text(-2, text_y_pos, 'MPC better', horizontalalignment='center', verticalalignment='bottom', color='black')
plt.arrow(1, text_y_pos, 2, 0, head_width=50, head_length=0.05, fc='black', ec='black')
plt.text(2, text_y_pos, 'Learned QP better', horizontalalignment='center', verticalalignment='bottom', color='black')

plt.xlabel('Ratio of penalized average cost (MPC / Learned QP) (log10)')
plt.ylim(0, y_max)

# %%
