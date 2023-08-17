import torch
from torch import nn
from torch.nn import functional as F
from torch.linalg import solve, inv, pinv
import numpy as np

from .preconditioner import Preconditioner
from utils.utils import bmv, bma

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
        """Specify P, H for fixed P, H, otherwise P, H needed to be given at forward.
        
        Assumes that H is full column rank when m >= n, and full row rank otherwise
        """
        super().__init__()
        self.device = device
        self.n = n
        self.m = m
        self.bP = torch.tensor(P, dtype=torch.float, device=device).unsqueeze(0) if P is not None else None       # (1, n, n)
        self.bH = torch.tensor(H, dtype=torch.float, device=device).unsqueeze(0) if H is not None else None       # (1, m, n)
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
        self.get_sol = self.get_sol_transform(self.bP, self.bH) if self.bP is not None and self.bH is not None else None

    def get_sol_transform(self, H, bP=None, bPinv=None):
        """Get the transform from z to x.
        
        Must specifying exactly one of bP or bPinv; specifying bPinv can reduce number of linear solves.
        """
        bH = self.bH if self.bH is not None else H
        if self.m >= self.n:
            return lambda z, q, b: bmv(pinv(bH), z - b)
        else:
            bP_param = bP if bP is not None else bPinv
            op = solve if bP is not None else bma
            def get_sol(z, q, b):
                t = lambda bM: bM.transpose(-1, -2)
                bPinvHt = op(bP_param, t(bH))
                Mt = solve(t(bH @ bPinvHt), t(bPinvHt))
                M = t(Mt)
                bPinvq = op(bP_param, q)
                return bmv(M @ bH, bPinvq) - bPinvq + bmv(M, z - b)
            return get_sol

    def get_AB(self, q, b, H=None, P=None, Pinv=None):
        """When P is not fixed, specifying exact one of P or Pinv; specifying bPinv can reduce number of linear solves."""
        # q: (bs, n), b: (bs, m)
        if self.bP is not None:
            bP_param = self.bP
            P_is_inv = False
        else:
            if P is not None:
                bP_param = P
                P_is_inv = False
            else:
                bP_param = Pinv
                P_is_inv = True
        op = solve if not P_is_inv else bma

        bH = self.bH if self.bH is not None else H
        D, tD = self.preconditioner(q, b, bP_param, H, input_P_is_inversed=P_is_inv, output_tD_is_inversed=False)   # (bs, m, m) or (1, m, m)
        mu = bmv(tD, bmv(bH, op(bP_param, q)) - b)  # (bs, m)
        tDD = tD @ D

        A = torch.cat([
            torch.cat([tDD, tD], 2),
            torch.cat([-2 * self.alpha * tDD + self.bIm, self.bIm - 2 * self.alpha * tD], 2),
        ], 1)   # (bs, 2m, 2m)
        B = torch.cat([
            mu,
            -2 * self.alpha * mu
        ], 1)   # (bs, 2m)
        return A, B

    def forward(self, q, b, P=None, H=None, Pinv=None, iters=1000):
        """
        Pass in one of P and Pinv when P is not specified as fixed. Using Pinv is more efficient in learned setting.
        """
        # q: (bs, n), b: (bs, m)
        bs = q.shape[0]
        if self.keep_X:
            Xs = torch.zeros((bs, iters + 1, 2 * self.m), device=self.device)
        else:
            Xs = None
        primal_sols = torch.zeros((bs, iters + 1, self.n), device=self.device)
        if self.warm_starter is not None:
            with torch.set_grad_enabled(self.is_warm_starter_trainable):
                qd, bd, Pd, Hd, Pinvd = map(lambda t: t.detach() if t is not None else None, [q, b, P, H, Pinv])
                P_param_to_ws = Pd if Pd is not None else Pinvd
                self.X0 = self.warm_starter(qd, bd, P_param_to_ws, Hd)
        get_sol = self.get_sol if self.get_sol is not None else self.get_sol_transform(H, P, Pinv)
        if self.keep_X:
            Xs[:, 0, :] = self.X0.clone()
        primal_sols[:, 0, :] = get_sol(self.X0[:, self.m:], q, b)
        X = self.X0
        A, B = self.get_AB(q, b, H, P, Pinv)
        for k in range(1, iters + 1):
            X = bmv(A, X) + B   # (bs, 2m)
            F.relu(X[:, self.m:], inplace=True)    # do projection
            if self.keep_X:
                Xs[:, k, :] = X.clone()
            primal_sols[:, k, :] = get_sol(X[:, self.m:], q, b)
        return Xs, primal_sols
