import torch
from torch import nn
from torch.nn import functional as F
from torch.linalg import solve, inv, pinv
import numpy as np

from .preconditioner import Preconditioner
from utils.utils import bmv

class QPSolver(nn.Module):
    """
    Solve QP problem:
    minimize    (1/2)x'Px + q'x
    subject to  Hx + b >= 0,
    where x in R^n, b in R^m.
    """
    def __init__(self, device, n, m,
            P=None, H=None,
            alpha=1, beta=1,
            preconditioner=None, warm_starter=None,
            is_warm_starter_trainable=False,
            keep_X=True):
        """Specify P, H for fixed P, H, otherwise P, H needed to be given at forward."""
        super().__init__()
        self.device = device
        self.n = n
        self.m = m
        self.bP = torch.tensor(P, dtype=torch.float, device=device).unsqueeze(0) if P is not None else None       # (1, n, n)
        self.bH = torch.tensor(H, dtype=torch.float, device=device).unsqueeze(0) if H is not None else None       # (1, m, n)
        self.bHinv = pinv(self.bH) if H is not None else None
        self.alpha = alpha
        self.beta = beta
        if preconditioner is None:
            # Use dummy preconditioner which gives D=I/beta
            self.preconditioner = Preconditioner(device, n, m, P, H, beta=beta, dummy=True)
        else:
            self.preconditioner = preconditioner
        self.warm_starter = warm_starter
        self.is_warm_starter_trainable = is_warm_starter_trainable
        self.keep_X = keep_X
        self.bIm = torch.eye(m, device=device).unsqueeze(0)
        self.X0 = torch.zeros((1, 2 * self.m), device=self.device)

    def get_AB(self, q, b, P=None, H=None):
        # q: (bs, n), b: (bs, m)
        bP = self.bP.broadcast_to((q.shape[0], -1, -1)) if self.bP is not None else P
        bH = self.bH if self.bH is not None else H
        D, tD = self.preconditioner(q, b, P, H)   # (bs, m, m) or (1, m, m)
        mu = bmv(tD, bmv(bH, solve(bP, q)) - b)  # (bs, m)

        A = torch.cat([
            torch.cat([tD @ D, tD], 2),
            torch.cat([-2 * self.alpha * tD @ D + self.bIm, self.bIm - 2 * self.alpha * tD], 2),
        ], 1)   # (bs, 2m, 2m)
        B = torch.cat([
            mu,
            -2 * self.alpha * mu
        ], 1)   # (bs, 2m)
        return A, B

    def forward(self, q, b, P=None, H=None, iters=1000):
        # q: (bs, n), b: (bs, m)
        bs = q.shape[0]
        if self.keep_X:
            Xs = torch.zeros((bs, iters + 1, 2 * self.m), device=self.device)
        else:
            Xs = None
        primal_sols = torch.zeros((bs, iters + 1, self.n), device=self.device)
        if self.warm_starter is not None:
            with torch.set_grad_enabled(self.is_warm_starter_trainable):
                self.X0 = self.warm_starter(q, b, P, H)
        bHinv = self.bHinv if self.bHinv is not None else pinv(bH)
        get_primal_sol = lambda X: bmv(bHinv, X[:, self.m:] - b)  # solve Hx + b = z
        if self.keep_X:
            Xs[:, 0, :] = self.X0.clone()
        primal_sols[:, 0, :] = get_primal_sol(self.X0)
        X = self.X0
        A, B = self.get_AB(q, b, P, H)
        for k in range(1, iters + 1):
            X = bmv(A, X) + B   # (bs, 2m)
            F.relu(X[:, self.m:], inplace=True)    # do projection
            if self.keep_X:
                Xs[:, k, :] = X.clone()
            primal_sols[:, k, :] = get_primal_sol(X)
        return Xs, primal_sols
