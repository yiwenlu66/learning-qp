import torch
import torch.nn as nn
import numpy as np
from utils import bmv

class LinearSystem():
    def __init__(self, A, B, Q, R, sqrt_W, x_min, x_max, u_min, u_max, bs, barrier_thresh, max_steps, device="cuda:0"):
        self.device = device
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.A = torch.tensor(A, dtype=torch.float, device=device).unsqueeze(0)
        self.B = torch.tensor(B, dtype=torch.float, device=device).unsqueeze(0)
        self.Q = torch.tensor(Q, dtype=torch.float, device=device).unsqueeze(0)
        self.R = torch.tensor(R, dtype=torch.float, device=device).unsqueeze(0)
        self.sqrt_W = torch.tensor(sqrt_W, dtype=torch.float, device=device).unsqueeze(0)
        self.x_min = torch.tensor(x_min, dtype=torch.float, device=device).unsqueeze(0)
        self.x_max = torch.tensor(x_max, dtype=torch.float, device=device).unsqueeze(0)
        self.u_min = torch.tensor(u_min, dtype=torch.float, device=device).unsqueeze(0)
        self.u_max = torch.tensor(u_max, dtype=torch.float, device=device).unsqueeze(0)
        self.bs = bs
        self.barrier_thresh = barrier_thresh
        self.max_steps = max_steps
        self.num_states = self.n
        self.num_actions = self.m
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n,))
        self.action_space = gym.spaces.Box(low=u_min, high=u_max, shape=(self.m,))
        self.state_space = self.observation_space
        self.x = torch.zeros((bs, self.n), device=device)
        self.x_ref = torch.zeros((bs, self.n), device=device)
        self.is_done = torch.zeros((bs,), dtype=torch.uint8, device=device)
        self.step_count = torch.zeros((bs,), dtype=torch.long, device=device)

    def obs(self):
        return self.x

    def reward(self):
        rew_main = -torch.dot(self.x, bmv(self.Q, self.x))
        rew_state_bar = torch.log((self.x_max - self.x) / self.barrier_thresh) + torch.log((self.x - self.x_min) / self.barrier_thresh)
        rew_done = -1.0 * (self.is_done == 1)
        return 10 + rew_main + rew_state_bar + 100 * rew_done


