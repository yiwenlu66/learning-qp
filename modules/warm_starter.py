import torch
from torch import nn
from torch.nn import functional as F
from torch.linalg import solve, inv, pinv
import numpy as np
from ..utils.torch_utils import vectorize_upper_triangular

class WarmStarter(nn.Module):
    def __init__(self, device, n, m, fixed_P=True, fixed_H=True):
        super().__init__()
        self.device = device
        self.n = n
        self.m = m
        self.fixed_P = fixed_P
        self.fixed_H = fixed_H
        num_in = n + m
        if not fixed_P:
            num_in += n * (n + 1) // 2
        if not fixed_H:
            num_in += n * m
        num_out = 2 * m
        num_hidden = max(num_in, num_out)
        self.net = nn.Sequential(
            nn.Linear(num_in, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_out),
        ).to(device=device)

    def forward(self, q, b, P=None, H=None):
        """The P argument can be either P or inv(P) in the original PDHG formulation, as long as consistent."""
        net_input = [q, b]
        if not self.fixed_P:
            net_input.append(vectorize_upper_triangular(P))
        if not self.fixed_H:
            net_input.append(H.flatten(start_dim=-2))
        net_input_t = torch.cat(net_input, 1)
        X = self.net(net_input_t)
        return X
