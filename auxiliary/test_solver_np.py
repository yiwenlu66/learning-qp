# %%
import numpy as np
import sys
import os
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, ".."))
from utils.osqp_utils import osqp_oracle

problem = np.load("example_problem.npy", allow_pickle=True).item()
q = problem["q"]
b = problem["b"]
P = problem["P"]
H = problem["H"]

def obj(x):
    return 0.5 * np.einsum('i,ij,j->', x, P, x) + np.dot(q, x)

x_star = osqp_oracle(q, b, P, H)

# %%
m, n = H.shape
D = np.eye(m)
Dt = np.linalg.inv(D + H @ np.linalg.solve(P, H.T))
mu = Dt @ (H @ np.linalg.solve(P, q) - b)
A = np.block([
    [Dt @ D, Dt],
    [-2 * Dt @ D + np.eye(m), np.eye(m) - 2 * Dt],
])
B = np.hstack([
    mu,
    -2 * mu
])
def iter(X):
    X = A @ X + B
    X[m:] = np.clip(X[m:], 0, np.inf)
    return X

# %%
def power_func(f, n):
    def helper(x):
        for _ in range(n):
            x = f(x)
        return x
    return helper

def get_sol(z):
    PinvHt = np.linalg.solve(P, H.T)
    M = np.linalg.solve((H @ PinvHt).T, PinvHt.T).T
    return -(np.eye(n) - M @ H) @ np.linalg.solve(P, q) + M @ (z - b)

X = power_func(iter, 10000)(np.zeros(2 * m))
z = X[m:]
x = get_sol(z)
obj(x)

# %%
