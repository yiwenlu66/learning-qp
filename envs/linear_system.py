import torch
import torch.nn as nn
import numpy as np
import random
import gym
from utils.utils import bmv, bqf

class LinearSystem():
    def __init__(self, A, B, Q, R, sqrt_W, x_min, x_max, u_min, u_max, bs, barrier_thresh, max_steps, device="cuda:0", random_seed=None, quiet=False, **kwargs):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
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
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2 * self.n,))
        self.action_space = gym.spaces.Box(low=u_min, high=u_max, shape=(self.m,))
        self.state_space = self.observation_space
        self.x = 0.5 * (self.x_max + self.x_min) * torch.ones((bs, self.n), device=device)
        self.u = torch.zeros((bs, self.m), device=device)
        self.x_ref = 0.5 * (self.x_max + self.x_min) * torch.ones((bs, self.n), device=device)
        self.is_done = torch.zeros((bs,), dtype=torch.uint8, device=device)
        self.step_count = torch.zeros((bs,), dtype=torch.long, device=device)
        self.quiet = quiet

    def obs(self):
        return torch.cat([self.x, self.x_ref], -1)

    def reward(self):
        rew_main = -bqf(self.x, self.Q) - bqf(self.u, self.R)
        rew_state_bar = torch.sum(torch.log(((self.x_max - self.x) / self.barrier_thresh).clamp(1e-8, 1.)) + torch.log(((self.x - self.x_min) / self.barrier_thresh).clamp(1e-8, 1.)), dim=-1)
        rew_done = -1.0 * (self.is_done == 1)
        if not self.quiet:
            print(rew_main.mean().item(), rew_state_bar.mean().item(), rew_done.mean().item())
        return 10 + 0.1 * rew_main + rew_state_bar + 10 * rew_done

    def done(self):
        return self.is_done.bool()

    def info(self):
        return {}

    def get_number_of_agents(self):
        return self.n

    def reset_done_envs(self, need_reset=None, x=None, x_ref=None):
        is_done = self.is_done.bool() if need_reset is None else need_reset
        size = torch.sum(is_done)
        self.step_count[is_done] = 0
        self.x_ref[is_done, :] = self.x_min + (self.x_max - self.x_min) * torch.rand((size, self.n), device=self.device) if x_ref is None else x_ref
        self.x[is_done, :] = self.x_min + (self.x_max - self.x_min) * torch.rand((size, self.n), device=self.device) if x is None else x
        self.is_done[is_done] = 0

    def reset(self, x=None, x_ref=None):
        self.reset_done_envs(torch.ones(self.bs, dtype=torch.bool, device=self.device))
        return self.obs()

    def check_in_bound(self):
        return ((self.x_min <= self.x) & (self.x <= self.x_max)).all(dim=-1)

    def step(self, u):
        self.reset_done_envs()
        self.u = u
        self.x = bmv(self.A, self.x) + bmv(self.B, u) + bmv(self.sqrt_W, torch.randn((self.bs, self.n), device=self.device))
        self.step_count += 1
        self.is_done[torch.logical_not(self.check_in_bound()).nonzero()] = 1   # 1 for failure
        self.is_done[self.step_count >= self.max_steps] = 2  # 2 for timeout
        return self.obs(), self.reward(), self.done(), self.info()

    def render(self, **kwargs):
        print(self.x, self.x_ref, self.u)
