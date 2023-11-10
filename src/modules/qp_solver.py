import torch
from torch import nn
from torch.nn import functional as F
from torch.linalg import solve, inv, pinv
import numpy as np

from .preconditioner import Preconditioner
from ..utils.torch_utils import bmv, bma, bsolve

class QPSolver(nn.Module):
    """
    Solve QP problem:
    minimize    (1/2)x'Px + q'x
    subject to  Hx + b >= 0,
    where x in R^n, b in R^m.
    """
    def __init__(self, device, n, m,
            P=None, Pinv=None, H=None,
            alpha=1, beta=1,
            preconditioner=None, warm_starter=None,
            is_warm_starter_trainable=False,
            keep_X=True,
            symmetric_constraint=False,
            buffered=False,
        ):
        """
        Initialize the QP solver.

        device: PyTorch device

        n, m: dimensions of decision variable x and constraint vector b

        P, Pinv, H: Optional matrices that define the QP. If not provided, must be supplied during forward pass. At most one of P and Pinv can be specified.

        alpha, beta: Parameters of the PDHG algorithm

        preconditioner: Optional preconditioner module

        warm_starter: Optional warm start module

        is_warm_starter_trainable: Flag for training the warm starter

        keep_X: Flag for keeping the primal-dual variable history

        symmetric_constraint: Flag for making the inequality constraint symmetric; when True, the constraint is assumed to be -1 <= Hx + b <= 1, instead of Hx + b >= 0.

        buffered: Flag for indicating whether the problem is modeled with the buffer variable \epsilon. When True, it is assumed that the first (n-1) decision variables are the original x, and the last decision variable is \epsilon; in this case, if symmetric constraint is enabled, then the projection is done as follows:
        1. Project epsilon to [0, +\infty)
        2. Project H_x x + b_x to [-1 - eps, 1 + eps]

        Note: Assumes that H is full column rank when m >= n, and full row rank otherwise.
        """
        super().__init__()
        self.device = device
        self.n = n
        self.m = m
        create_tensor = lambda t: (torch.tensor(t, dtype=torch.float, device=device).unsqueeze(0) if t is not None else None) if type(t) != torch.Tensor else t.unsqueeze(0)
        assert (P is None) or (Pinv is None), "At most one of P and Pinv can be specified"
        self.bP = create_tensor(P)       # (1, n, n)
        self.bPinv = create_tensor(Pinv)       # (1, n, n)
        self.bH = create_tensor(H)       # (1, m, n)
        self.alpha = alpha
        self.beta = beta
        if preconditioner is None:
            # Use dummy preconditioner which gives D=I/beta
            self.preconditioner = Preconditioner(device, n, m, P=P, Pinv=Pinv, H=H, beta=beta, dummy=True)
        else:
            self.preconditioner = preconditioner
        self.warm_starter = warm_starter
        self.is_warm_starter_trainable = is_warm_starter_trainable
        self.keep_X = keep_X
        self.symmetric_constraint = symmetric_constraint
        self.buffered = buffered

        self.bIm = torch.eye(m, device=device).unsqueeze(0)
        self.X0 = torch.zeros((1, 2 * self.m), device=self.device)

        # If P, H are constant, we can pre-compute the transformation from z to x
        if self.bP is not None and self.bH is not None:
            self.get_sol = self.get_sol_transform(self.bP, self.bH)
        elif self.bPinv is not None and self.bH is not None:
            self.get_sol = self.get_sol_transform(self.bH, bPinv=self.bPinv)
        else:
            self.get_sol = None

        # If possible, cache intermediate results in the computation of the affine transform used for each PDHG iteration
        if (P is not None or Pinv is not None) and H is not None and preconditioner is None:
            self.cache_keys = ["D", "tD", "tDD", "A"]
        else:
            self.cache_keys = []
        self.cache = {}

    def get_sol_transform(self, H, bP=None, bPinv=None):
        """
        Computes the transformation from dual variable z to primal variable x.

        H: Constraint matrix
        bP, bPinv: Either the matrix P or its inverse. Exactly one must be specified. Specifying Pinv can reduce number of linear solves.

        Returns: Function that performs the transformation
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
        """
        Computes matrices A and B used in the PDHG iterations.

        q, b: Coefficients in the objective and constraint
        H, P, Pinv: Matrix H, and (either the matrix P or its inverse). Must be specified if not initialized. Specifying Pinv can reduce number of linear solves.

        Returns: Matrices A and B
        """

        def _lookup_or_compute(keys, compute_fn):
            """Lookup variable(s) from cache or compute them if not available.

            keys: either a variable name (str), or a list of variable names
            compute_fn: function that computes the variable(s) if not available in cache; returns a single value if keys is a string, or a tuple of values if keys is a list
            """
            is_single = (type(keys) == str)
            if is_single:
                keys = [keys]
            if not all([key in self.cache for key in keys]):
                values = compute_fn()
                if is_single:
                    values = (values,)
                for key, value in zip(keys, values):
                    if key in self.cache_keys:
                        self.cache[key] = value
            else:
                values = tuple([self.cache[key] for key in keys])
            return values if not is_single else values[0]

        # q: (bs, n), b: (bs, m)
        if self.bP is not None or self.bPinv is not None:
            if self.bP is not None:
                bP_param = self.bP
                P_is_inv = False
            else:
                bP_param = self.bPinv
                P_is_inv = True
        else:
            if P is not None:
                bP_param = P
                P_is_inv = False
            else:
                bP_param = Pinv
                P_is_inv = True
        op = bsolve if not P_is_inv else bma

        bH = self.bH if self.bH is not None else H
        D, tD = _lookup_or_compute(["D", "tD"], lambda: self.preconditioner(q, b, bP_param, H, input_P_is_inversed=P_is_inv, output_tD_is_inversed=False))   # (bs, m, m) or (1, m, m)
        mu = bmv(tD, bmv(bH, op(bP_param, q)) - b)  # (bs, m)
        tDD = _lookup_or_compute("tDD", lambda: tD @ D)

        A = _lookup_or_compute("A", lambda:
            torch.cat([
                torch.cat([tDD, tD], 2),
                torch.cat([-2 * self.alpha * tDD + self.bIm, self.bIm - 2 * self.alpha * tD], 2),
            ], 1)   # (bs, 2m, 2m)
        )
        B = torch.cat([
            mu,
            -2 * self.alpha * mu
        ], 1)   # (bs, 2m)
        return A, B

    def compute_residuals(self, x, z, u, q, b, P=None, H=None, Pinv=None):
        """
        Computes the primal and dual residuals.

        x, z: Primal variables
        u: Dual variable
        q, b: Coefficients in the objective and constraint
        P, H, Pinv: Optional matrices defining the QP. Must be provided if not initialized.

        Returns: Primal and dual residuals
        """
        # Determine effective P and H matrices
        if self.bP is not None or self.bPinv is not None:
            if self.bP is not None:
                eff_P = self.bP
                P_is_inv = False
            else:
                eff_P = self.bPinv
                P_is_inv = True
        else:
            if P is not None:
                eff_P = P
                P_is_inv = False
            else:
                eff_P = Pinv
                P_is_inv = True

        if self.bH is not None:
            eff_H = self.bH
        else:
            eff_H = H

        # Compute primal residual: Hx + b - z
        primal_residual = bmv(eff_H, x) + b - z

        # Determine the operation for multiplying with P or its inverse
        op = bsolve if P_is_inv else bmv

        # Compute dual residual: Px + q + H'u
        dual_residual = op(eff_P, x) + q + bmv(eff_H.transpose(-1, -2), u)

        return primal_residual, dual_residual


    def forward(
        self, q, b,
        P=None, H=None, Pinv=None,
        iters=1000,
        only_last_primal=True,
        return_residuals=False
    ):
        """
        Solves the QP problem using PDHG.

        q, b: Coefficients in the objective and constraint
        P, H, Pinv: Optional matrices defining the QP, i.e., matrix H, and (either the matrix P or its inverse). Must be provided if not initialized. Using Pinv is more efficient in learned setting.
        iters: Number of PDHG iterations
        only_last_primal: Flag for returning only the last primal solution (when True, primal_sols is (bs, 1, n); otherwise (bs, iters + 1, n))
        return_residuals: Flag for returning residuals

        Returns: History of primal-dual variables, primal solutions, and optionally residuals of the last iteration
        """
        # q: (bs, n), b: (bs, m)
        bs = q.shape[0]
        if self.keep_X:
            Xs = torch.zeros((bs, iters + 1, 2 * self.m), device=self.device)
        else:
            Xs = None
        primal_sols = torch.zeros((bs, (iters if not only_last_primal else 0) + 1, self.n), device=self.device)
        if self.warm_starter is not None:
            with torch.set_grad_enabled(self.is_warm_starter_trainable):
                qd, bd, Pd, Hd, Pinvd = map(lambda t: t.detach() if t is not None else None, [q, b, P, H, Pinv])
                P_param_to_ws = Pd if Pd is not None else Pinvd
                self.X0 = self.warm_starter(qd, bd, P_param_to_ws, Hd)
        get_sol = self.get_sol if self.get_sol is not None else self.get_sol_transform(H, P, Pinv)
        if self.keep_X:
            Xs[:, 0, :] = self.X0.clone()
        if not only_last_primal:
            primal_sols[:, 0, :] = get_sol(self.X0[:, self.m:], q, b)
        X = self.X0
        A, B = self.get_AB(q, b, H, P, Pinv)
        for k in range(1, iters + 1):
            # PDHG update
            X = bmv(A, X) + B   # (bs, 2m)
            if not self.symmetric_constraint:
                # Project to [0, +\infty)
                F.relu(X[:, self.m:], inplace=True)
            else:
                if not self.buffered:
                    # Project to [-1, 1]
                    projected = torch.clamp(X[:, self.m:], -1, 1)
                    X = torch.cat((X[:, :self.m], projected), dim=1)
                else:
                    # Hybrid projection: epsilon to [0, +\infty), the rest decision variables to [-1 - eps, 1 + eps]
                    # Project epsilon
                    F.relu(X[:, -1:], inplace=True)
                    # Project the rest variables
                    projected = torch.clamp(X[:, self.m:-1], -1 - X[:, -1:], 1 + X[:, -1:])
                    # Concatenate
                    X = torch.cat((X[:, :self.m], projected, X[:, -1:]), dim=1)
            if self.keep_X:
                Xs[:, k, :] = X.clone()
            if not only_last_primal:
                primal_sols[:, k, :] = get_sol(X[:, self.m:], q, b)

        if only_last_primal:
            primal_sols[:, 0, :] = get_sol(X[:, self.m:], q, b)

        # Compute residuals for the last step if the flag is set
        if return_residuals:
            x_last = primal_sols[:, -1, :]
            z_last = Xs[:, -1, self.m:]
            u_last = Xs[:, -1, :self.m]
            primal_residual, dual_residual = self.compute_residuals(x_last, z_last, u_last, q, b, P, H, Pinv)
            return Xs, primal_sols, (primal_residual, dual_residual)
        else:
            return Xs, primal_sols
