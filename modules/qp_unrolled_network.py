import torch
from torch import nn
import numpy as np
import scipy
from modules.qp_solver import QPSolver
from modules.warm_starter import WarmStarter
from utils.utils import make_psd, interpolate_state_dicts, mpc2qp, osqp_oracle, np_batch_op


class QPUnrolledNetwork(nn.Module):
    """
    Learn a QP problem from the input using a MLP, then solve the QP using fixed number of unrolled PDHG iterations.

    Form of QP:
    minimize    (1/2)x'Px + q'x
    subject to  Hx + b >= 0,
    where x in R^n, b in R^m.
    """
    def __init__(
        self, device, input_size, n_qp, m_qp, qp_iter, mlp_builder,
        shared_PH=False,
        affine_qb=False,
        use_warm_starter=False,
        train_warm_starter=False,
        ws_loss_coef=1.,
        ws_update_rate=0.01,
        ws_loss_shaper=lambda x: x ** (1 / 2),
        mpc_baseline=None,
        use_osqp_for_mpc=False,
        use_residual_loss=False,
    ):
        """mlp_builder is a function mapping (input_size, output_size) to a nn.Sequential object.
        
        If shared_PH == True, P and H are parameters indepedent of input, and q and b are functions of input;
        Otherwise, (P, H, q, b) are all functions of input.

        If affine_qb == True, then q and b are restricted to be affine functions of input.
        """

        super().__init__()

        self.shared_PH = shared_PH
        self.affine_qb = affine_qb
        self.device = device
        self.input_size = input_size
        self.n_qp = n_qp
        self.m_qp = m_qp
        self.qp_iter = qp_iter

        self.n_P_param = n_qp * (n_qp + 1) // 2
        self.n_q_param = n_qp
        self.n_H_param = m_qp * n_qp
        self.n_b_param = m_qp

        self.n_mlp_output = 0
        if not self.shared_PH:
            self.n_mlp_output += (self.n_P_param + self.n_H_param)
            self.P_params = None
            self.H_params = None
        else:
            self.P_params = nn.Parameter(torch.randn((self.n_P_param,), device=device)) 
            self.H_params = nn.Parameter(torch.randn((self.n_H_param,), device=device)) 

        if not self.affine_qb:
            self.n_mlp_output += (self.n_q_param + self.n_b_param)
            self.qb_affine_layer = None
        else:
            self.qb_affine_layer = nn.Linear(input_size, self.n_q_param + self.n_b_param)

        if self.n_mlp_output > 0:
            self.mlp = mlp_builder(input_size, self.n_mlp_output)
        else:
            self.mlp = None

        # TODO: add preconditioner
        self.warm_starter = WarmStarter(device, n_qp, m_qp, fixed_P=shared_PH, fixed_H=shared_PH) if use_warm_starter else None
        self.warm_starter_delayed = WarmStarter(device, n_qp, m_qp, fixed_P=shared_PH, fixed_H=shared_PH) if use_warm_starter else None
        self.train_warm_starter = train_warm_starter
        self.ws_loss_coef = ws_loss_coef
        self.ws_update_rate = ws_update_rate
        self.ws_loss_shaper = ws_loss_shaper

        # is_warm_starter_trainable is always False, since the warm starter is trained via another inference independent of the solver
        self.solver = QPSolver(device, n_qp, m_qp, warm_starter=self.warm_starter_delayed, is_warm_starter_trainable=False)

        # Includes losses generated by the model itself (indepedent of interaction with env), i.e., warm starting & preconditioning
        self.autonomous_losses = {}

        self.mpc_baseline = mpc_baseline
        self.use_osqp_for_mpc = use_osqp_for_mpc

        # Whether to consider residual loss during training - this can encourage feasibility of the learned QP problem
        self.use_residual_loss = use_residual_loss

    def compute_warm_starter_loss(self, q, b, Pinv, H, solver_Xs):
        qd, bd, Pinvd, Hd = map(lambda t: t.detach() if t is not None else None, [q, b, Pinv, H])
        X0 = self.warm_starter(qd, bd, Pinvd, Hd)
        gt = solver_Xs[:, -1, :].detach()
        return self.ws_loss_coef * self.ws_loss_shaper(((gt - X0) ** 2).sum(dim=-1).mean())

    def run_mpc_baseline(self, x, use_osqp_oracle=False):
        t = lambda a: torch.tensor(a, device=x.device, dtype=torch.float)
        n, m, P, q, H, b = mpc2qp(
            self.mpc_baseline["n_mpc"],
            self.mpc_baseline["m_mpc"],
            self.mpc_baseline["N"],
            t(self.mpc_baseline["A"]),
            t(self.mpc_baseline["B"]),
            t(self.mpc_baseline["Q"]),
            t(self.mpc_baseline["R"]),
            self.mpc_baseline["x_min"],
            self.mpc_baseline["x_max"],
            self.mpc_baseline["u_min"],
            self.mpc_baseline["u_max"],
            *self.mpc_baseline["obs_to_state_and_ref"](x),
            normalize=self.mpc_baseline.get("normalize", False),
        )
        if not use_osqp_oracle:
            solver = QPSolver(x.device, n, m, P, H)
            Xs, primal_sols = solver(q, b, iters=100)
            sol = primal_sols[:, -1, :]
        else:
            f = lambda t: t.detach().cpu().numpy()
            f_sparse = lambda t: scipy.sparse.csc_matrix(t.cpu().numpy())
            t = lambda a: torch.tensor(a, dtype=torch.float, device=self.device)
            if q.shape[0] > 1:
                sol = t(np_batch_op(osqp_oracle, f(q), f(b), f_sparse(P), f_sparse(H)))
            else:
                sol = t(osqp_oracle(f(q[0, :]), f(b[0, :]), f_sparse(P), f_sparse(H))).unsqueeze(0)
        return sol, (P.unsqueeze(0), q, H.unsqueeze(0), b)

    def forward(self, x, return_problem_params=False):
        if self.mpc_baseline is not None:
            sol, problem_params = self.run_mpc_baseline(x, use_osqp_oracle=self.use_osqp_for_mpc)
        else:
            bs = x.shape[0]
            if self.mlp is not None:
                qp_params = self.mlp(x)

            # Decode MLP output
            end = 0
            if not self.shared_PH:
                start = end
                end = start + self.n_P_param
                P_params = qp_params[:, start:end]
                start = end
                end = start + self.n_H_param
                H_params = qp_params[:, start:end]
            else:
                P_params = self.P_params.unsqueeze(0)
                H_params = self.H_params.unsqueeze(0)

            if not self.affine_qb:
                start = end
                end = start + self.n_q_param
                q = qp_params[:, start:end]
                start = end
                end = start + self.n_b_param
                b = qp_params[:, start:end]
            else:
                q_b_params = self.qb_affine_layer(x)
                q = q_b_params[:, :self.n_q_param]
                b = q_b_params[:, self.n_q_param:]

            # Reshape P, H vectors into matrices
            Pinv = make_psd(P_params, min_eig=1e-2)
            H = H_params.view(-1, self.m_qp, self.n_qp)

            # Update parameters of warm starter with a delay to stabilize training
            if self.train_warm_starter:
                self.warm_starter_delayed.load_state_dict(interpolate_state_dicts(self.warm_starter_delayed.state_dict(), self.warm_starter.state_dict(), self.ws_update_rate))

            if self.use_residual_loss:
                Xs, primal_sols, residuals = self.solver(q, b, Pinv=Pinv, H=H, iters=self.qp_iter, return_residuals=True)
                primal_residual, dual_residual = residuals
                residual_loss = ((primal_residual ** 2).sum(dim=-1) + (dual_residual ** 2).sum(dim=-1)).mean()
                self.autonomous_losses["residual"] = 1e-3 * residual_loss
            else:
                Xs, primal_sols = self.solver(q, b, Pinv=Pinv, H=H, iters=self.qp_iter)
            if self.train_warm_starter:
                self.autonomous_losses["warm_starter"] = self.compute_warm_starter_loss(q, b, Pinv, H, Xs)
            sol = primal_sols[:, -1, :]
            if return_problem_params:
                problem_params = (torch.linalg.inv(Pinv), q, H, b)

        if not return_problem_params:
            # Only return the solution
            return sol
        else:
            # Return the solution as well as (P, q, H, b)
            return sol, problem_params
