import torch
from torch import nn
from torch.nn import functional as F
from torch.linalg import solve, inv, pinv
import numpy as np

from utils.utils import make_psd, vectorize_upper_triangular

class Preconditioner(nn.Module):
    def __init__(self, device, n, m,
            P=None, H=None,
            dummy=False,
            beta=1,
            adaptive=False):
        """
        dummy = True: fix D = I
        adaptive = False: use same D for all q, b; adaptive = True: determine D based on q, b
        Specify P, H if they are fixed; otherwise they need to be passed in when calling forward.
        """
        super().__init__()
        self.device = device
        self.n = n
        self.m = m
        self.P = torch.tensor(P, dtype=torch.float, device=device) if P is not None else None       # (1, n, n)
        self.H = torch.tensor(H, dtype=torch.float, device=device) if H is not None else None       # (1, m, n)
        self.dummy = dummy
        self.beta = beta
        self.adaptive = adaptive
        self.bP = self.P.unsqueeze(0) if P is not None else None
        self.bH = self.H.unsqueeze(0) if H is not None else None
        self.bHPinvHt = (self.H @ solve(self.P, self.H.t())).unsqueeze(0) if P is not None and H is not None else None   # (1, m, m)

        # Parameterize D using Cholesky decomposition
        num_param = m * (m + 1) // 2
        if not dummy:
            if not adaptive:
                self.param = nn.Parameter(torch.zeros((num_param,), device=device))  # (m, m)
            else:
                num_in = n + m
                if self.bP is None:
                    num_in += n * (n + 1) // 2
                if self.bH is None:
                    num_in += n * m
                self.D_net = nn.Sequential(
                    nn.Linear(num_in, num_in),
                    nn.ReLU(),
                    nn.Linear(num_in, num_param),
                ).to(device=device)

    def forward(self, q=None, b=None, P=None, H=None):
        # q: (bs, n), b: (bs, m)
        if self.dummy:
            D = torch.eye(self.m, device=self.device)
        elif not self.adaptive:
            D = make_psd(self.param.unsqueeze(0))   # (1, m, m)
        else:
            assert q is not None and b is not None
            net_input = [q, b]
            if self.bP is None:
                net_input.append(vectorize_upper_triangular(P))
            if self.bH is None:
                net_input.append(H.flatten(start_dim=-2))
            net_input_t = torch.cat(net_input, 1) * 1e-6
            D = make_psd(self.D_net(net_input_t))        # (bs, m, m)
        D /= self.beta
        bH = self.bH if self.bH is not None else H
        bP = self.bP if self.bP is not None else P
        bHPinvHt = self.bHPinvHt if self.bHPinvHt is not None else (bH @ solve(bP, bH.transpose(-1, -2)))
        tD = inv(D + bHPinvHt)   # (*, m, m)
        return D, tD
