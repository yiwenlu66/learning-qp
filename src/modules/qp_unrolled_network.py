import torch
from torch import nn
import numpy as np
import scipy
import functools
from ..modules.qp_solver import QPSolver
from ..modules.warm_starter import WarmStarter
from ..utils.torch_utils import make_psd, interpolate_state_dicts
from ..utils.mpc_utils import mpc2qp, scenario_robust_mpc, tube_robust_mpc
from ..utils.osqp_utils import osqp_oracle
from ..utils.np_batch_op import np_batch_op
import os
from concurrent.futures import ThreadPoolExecutor


class StrictAffineLayer(nn.Module):
    """
    Layer mapping from obs to (q, b) in the strict affine form.
    """
    def __init__(self, input_size, n, m, obs_has_half_ref):
        super().__init__()
        self.obs_has_half_ref = obs_has_half_ref
        self.input_size = input_size
        self.q_layer = nn.Linear(self.input_size, n, bias=False)
        if not self.obs_has_half_ref:
            self.b_layer = nn.Linear(self.input_size // 2, m, bias=True)
        else:
            self.b_layer = nn.Linear(self.input_size, m, bias=True)

    def forward(self, x):
        if not self.obs_has_half_ref:
            x0 = x[:, :self.input_size // 2]
        else:
            x0 = x
        q = self.q_layer(x)
        b = self.b_layer(x0)
        return torch.cat([q, b], dim=1)


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
        strict_affine_layer=False,
        obs_has_half_ref=False,
        symmetric=False,
        no_b=False,
        use_warm_starter=False,
        train_warm_starter=False,
        ws_loss_coef=1.,
        ws_update_rate=0.01,
        ws_loss_shaper=lambda x: x ** (1 / 2),
        mpc_baseline=None,
        use_osqp_for_mpc=False,
        imitate_mpc=False,
        use_residual_loss=False,
        force_feasible=False,
        feasible_lambda=10,
        is_test=False,
    ):
        """mlp_builder is a function mapping (input_size, output_size) to a nn.Sequential object.

        If shared_PH == True, P and H are parameters indepedent of input, and q and b are functions of input;
        Otherwise, (P, H, q, b) are all functions of input.

        If affine_qb == True, then q and b are restricted to be affine functions of input.

        If strict_affine_layer == True (only effective when affine_qb=True), then:
        1. q is linear w.r.t. (x0, xref) (no bias)
        2. b is affine w.r.t. x0 (no dependence on xref)

        If obs_has_half_ref == True, the policy knows that the observation is in the form (x0, xref), with each taking up half of the dimension of the observation.

        If symmetric == True (only effective when affine_qb=True), then:
        1. The bias terms are disabled in the modeling of q and b, i.e., q = Wq * x, b = Wb * x.
        2. The constraint is assumed to be -1 <= Hx + b <= 1, instead of Hx + b >= 0.

        If no_b == True in addition to symmetric == True, then b is skipped altogether, i.e., the constraint is assumed to be -1 <= Hx <= 1.

        If mpc_baseline != None and imitate_mpc == False, then the forward function directly returns the solution of the MPC problem, instead of solving the learned QP problem. Can be used for benchmarking MPC.

        If mpc_baseline != None and imitate_mpc == True, then the forward function returns the solution of the learned QP problem, but a loss term is computed using the MPC problem. Can be used for supervised imitation learning.

        If force_feasible == True, solve the following problem instead of the original QP problem:
        minimize_{x,y}    (1/2)x'Px + q'x + lambda * y^2
        s.t.       Hx + b + y * 1 >= 0, y >= 0,
        where x in R^n, y in R.
        In this case, the solution returned will be of dimension (n + 1).
        """

        super().__init__()

        self.shared_PH = shared_PH
        self.affine_qb = affine_qb
        self.strict_affine_layer = strict_affine_layer
        self.obs_has_half_ref = obs_has_half_ref

        self.device = device
        self.input_size = input_size

        # QP dimensions: there are the number of variables and constraints WITHOUT considering the slack variable
        self.n_qp = n_qp
        self.m_qp = m_qp

        self.qp_iter = qp_iter

        self.symmetric = symmetric
        self.no_b = no_b

        self.n_P_param = n_qp * (n_qp + 1) // 2
        self.n_q_param = n_qp
        self.n_H_param = m_qp * n_qp
        self.n_b_param = m_qp if not self.no_b else 0

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
            if not self.strict_affine_layer:
                self.qb_affine_layer = nn.Linear(input_size, self.n_q_param + self.n_b_param, bias=not self.symmetric)
            else:
                self.qb_affine_layer = StrictAffineLayer(input_size, self.n_qp, self.m_qp, self.obs_has_half_ref)

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

        # P, H are fixed when the model is in test mode, and they are constant across all states (i.e., shared_PH == True)
        self.fixed_PH = is_test and shared_PH

        # Includes losses generated by the model itself (indepedent of interaction with env), e.g., warm starting & preconditioning
        self.autonomous_losses = {}

        self.mpc_baseline = mpc_baseline
        self.use_osqp_for_mpc = use_osqp_for_mpc

        self.imitate_mpc = imitate_mpc

        # Whether to consider residual loss during training - this can encourage feasibility of the learned QP problem
        self.use_residual_loss = use_residual_loss

        # Whether to force the problem to be feasible
        self.force_feasible = force_feasible
        self.feasible_lambda = feasible_lambda

        self.solver = None

        self.info = {}

        # Reserved for storing the controllers for each simulation instance when robust MPC is enabled
        self.robust_controllers = []

        # Store info returned by env
        self.env_info = {}

        # When running batch testing, mask envs already done, to speed up computation (implemented for robust mpc); initialized at inference time since batch size is not known during initialization
        self.is_active = None


    def initialize_solver(self):
        # If the problem is forced to be feasible, the dimension of the solution is increased by 1 (introduce slack variable)
        n_qp_actual = self.n_qp + 1 if self.force_feasible else self.n_qp
        m_qp_actual = self.m_qp + 1 if self.force_feasible else self.m_qp

        # is_warm_starter_trainable is always False, since the warm starter is trained via another inference independent of the solver
        # When self.fixed_PH == True, the solver is initialized with fixed P, H matrices; otherwise, P, H are not passed to the solver during initialization time, but computed during the forward pass instead
        if not self.fixed_PH:
            self.solver = QPSolver(self.device, n_qp_actual, m_qp_actual, warm_starter=self.warm_starter_delayed, is_warm_starter_trainable=False, symmetric_constraint=self.symmetric, buffered=self.force_feasible)
        else:
            # Should be called after loading state dict
            Pinv, H = self.get_PH()
            self.solver = QPSolver(self.device, n_qp_actual, m_qp_actual, Pinv=Pinv.squeeze(0), H=H.squeeze(0), warm_starter=self.warm_starter_delayed, is_warm_starter_trainable=False, symmetric_constraint=self.symmetric, buffered=self.force_feasible)

    def compute_warm_starter_loss(self, q, b, Pinv, H, solver_Xs):
        qd, bd, Pinvd, Hd = map(lambda t: t.detach() if t is not None else None, [q, b, Pinv, H])
        X0 = self.warm_starter(qd, bd, Pinvd, Hd)
        gt = solver_Xs[:, -1, :].detach()
        return self.ws_loss_coef * self.ws_loss_shaper(((gt - X0) ** 2).sum(dim=-1).mean())

    def parallel_controller_creation(self, controller_creator, xref_np, bs):
        """
        Create robust MPC controlller in parallel
        """
        # Helper function for parallel execution
        def task_creator(index):
            return controller_creator(self.mpc_baseline, xref_np[index, :])

        with ThreadPoolExecutor() as executor:
            # Executing the tasks in parallel
            results = executor.map(task_creator, range(bs))

        # Collecting the results
        self.robust_controllers.extend(results)

    def run_mpc_baseline(self, x, use_osqp_oracle=False):
        robust_method = self.mpc_baseline.get("robust_method", None)
        x0, xref = self.mpc_baseline["obs_to_state_and_ref"](x)
        bs = x.shape[0]

        # Conversions between torch and np
        t = lambda a: torch.tensor(a, device=x.device, dtype=torch.float)
        f = lambda t: t.detach().cpu().numpy()
        f_sparse = lambda t: scipy.sparse.csc_matrix(t.cpu().numpy())

        if robust_method is None:
            # Run vanilla MPC without robustness
            eps = 1e-3
            n, m, P, q, H, b = mpc2qp(
                self.mpc_baseline["n_mpc"],
                self.mpc_baseline["m_mpc"],
                self.mpc_baseline["N"],
                t(self.mpc_baseline["A"]),
                t(self.mpc_baseline["B"]),
                t(self.mpc_baseline["Q"]),
                t(self.mpc_baseline["R"]),
                self.mpc_baseline["x_min"] + eps,
                self.mpc_baseline["x_max"] - eps,
                self.mpc_baseline["u_min"],
                self.mpc_baseline["u_max"],
                x0,
                xref,
                normalize=self.mpc_baseline.get("normalize", False),
                Qf=self.mpc_baseline.get("terminal_coef", 0.) * t(np.eye(self.mpc_baseline["n_mpc"])) if self.mpc_baseline.get("Qf", None) is None else t(self.mpc_baseline["Qf"]),
            )
            if not use_osqp_oracle:
                solver = QPSolver(x.device, n, m, P=P, H=H)
                Xs, primal_sols = solver(q, b, iters=100)
                sol = primal_sols[:, -1, :]
            else:
                osqp_oracle_with_iter_count = functools.partial(osqp_oracle, return_iter_count=True)
                if q.shape[0] > 1:
                    sol_np, iter_counts = np_batch_op(osqp_oracle_with_iter_count, f(q), f(b), f_sparse(P), f_sparse(H))
                    sol = t(sol_np)
                else:
                    sol_np, iter_count = osqp_oracle_with_iter_count(f(q[0, :]), f(b[0, :]), f_sparse(P), f_sparse(H))
                    sol = t(sol_np).unsqueeze(0)
                    iter_counts = np.array([iter_count])
                # Save OSQP iteration counts into the info dict
                if "osqp_iter_counts" not in self.info:
                    self.info["osqp_iter_counts"] = iter_counts
                else:
                    self.info["osqp_iter_counts"] = np.concatenate([self.info["osqp_iter_counts"], iter_counts])
            return sol, (P.unsqueeze(0), q, H.unsqueeze(0), b)

        elif robust_method in ["scenario", "tube"]:
            # Set up scenario or tube MPC
            if not self.robust_controllers:
                # Create a controller for each simulation instance, according to the current reference (note: this assumes that the mapping from instance index to reference is constant)
                controller_creator = {
                    "scenario": scenario_robust_mpc,
                    "tube": tube_robust_mpc,
                }[robust_method]
                xref_np = f(xref)
                self.parallel_controller_creation(controller_creator, xref_np, bs)
                self.is_active = np.ones((bs,), dtype=bool)

            # Get solutions according to current state
            x0_np = f(x0)
            already_on_stats = f(self.env_info.get("already_on_stats", torch.zeros((bs,), dtype=bool))).astype(bool)
            self.is_active = np.logical_not(already_on_stats) & self.is_active   # Skip computation for instances already done
            get_solution = lambda i: self.robust_controllers[i](x0_np[i, :], is_active=self.is_active[i])
            sol_np, running_time = np_batch_op(get_solution, np.arange(bs))
            sol = t(sol_np)

            # Save running time to info dict
            non_zero_mask = running_time != 0.  # Filter out instances that are already done
            running_time_eff = running_time[non_zero_mask]
            if "running_time" not in self.info:
                self.info["running_time"] = running_time_eff
            else:
                self.info["running_time"] = np.concatenate([self.info["running_time"], running_time_eff])

            return sol, None


    def get_PH(self, mlp_out=None):
        """
        Compute P, H matrices from the parameters.
        Notice: returns (Pinv, H) instead of (P, H)
        """
        # Decode MLP output
        end = 0
        if not self.shared_PH:
            start = end
            end = start + self.n_P_param
            P_params = mlp_out[:, start:end]
            start = end
            end = start + self.n_H_param
            H_params = mlp_out[:, start:end]
        else:
            P_params = self.P_params.unsqueeze(0)
            H_params = self.H_params.unsqueeze(0)

        # Reshape P, H vectors into matrices
        Pinv = make_psd(P_params, min_eig=1e-2)
        H = H_params.view(-1, self.m_qp, self.n_qp)

        # If the problem is forced to be feasible, compute the parameters (\tilde{P}, \tilde{H}) of the augmented problem
        # \tilde{P} = [P, 0; 0, lambda]
        if self.force_feasible:
            zeros_n = torch.zeros((1, self.n_qp, 1), device=self.device)
            I = torch.eye(1, device=self.device).unsqueeze(0)
            tilde_P_inv = torch.cat([
                torch.cat([Pinv, zeros_n], dim=2),
                torch.cat([zeros_n.transpose(1, 2), 1 / self.feasible_lambda * I], dim=2)
            ], dim=1)
            # \tilde{H} = [H, I; 0, I]
            ones_m = torch.ones((1, self.m_qp, 1), device=self.device)
            tilde_H = torch.cat([
                torch.cat([H, ones_m], dim=2),
                torch.cat([zeros_n.transpose(1, 2), I], dim=2)
            ], dim=1)
            Pinv, H = tilde_P_inv, tilde_H
        return Pinv, H

    def get_qb(self, x, mlp_out=None):
        """
        Compute q, b vectors from the parameters.
        """
        bs = x.shape[0]
        end = self.n_P_param + self.n_H_param if not self.shared_PH else 0
        if not self.affine_qb:
            start = end
            end = start + self.n_q_param
            q = mlp_out[:, start:end]
            start = end
            end = start + self.n_b_param
            b = mlp_out[:, start:end]
        else:
            qb = self.qb_affine_layer(x)
            q = qb[:, :self.n_q_param]
            b = qb[:, self.n_q_param:]
        if self.no_b:
            b = torch.zeros((bs, self.m_qp), device=self.device)

        # If the problem is forced to be feasible, compute the parameters (\tilde{q}, \tilde{b}) of the augmented problem
        if self.force_feasible:
            zeros_1 = torch.zeros((bs, 1), device=self.device)
            # \tilde{q} = [q; 0]
            tilde_q = torch.cat([q, zeros_1], dim=1)
            # \tilde{b} = [b; 0]
            tilde_b = torch.cat([b, zeros_1], dim=1)
            q, b = tilde_q, tilde_b

        return q, b

    def forward(self, x, return_problem_params=False, info=None):
        if info is not None:
            self.env_info = info
        if self.mpc_baseline is not None:
            mpc_sol, mpc_problem_params = self.run_mpc_baseline(x, use_osqp_oracle=self.use_osqp_for_mpc)

        if (self.mpc_baseline is not None) and (not self.imitate_mpc):
            # MPC solution is directly used as the final solution
            sol, problem_params = mpc_sol, mpc_problem_params
        else:
            # Check whether solver has been initialized
            if self.solver is None:
                self.initialize_solver()

            bs = x.shape[0]

            # Run MLP forward pass, if necessary
            if self.mlp is not None:
                mlp_out = self.mlp(x)
            else:
                mlp_out = None

            # Compute P, H, if they are not fixed
            if not self.fixed_PH:
                Pinv, H = self.get_PH(mlp_out)
            else:
                Pinv, H = None, None

            # Compute q, b
            q, b = self.get_qb(x, mlp_out)

            # Update parameters of warm starter with a delay to stabilize training
            if self.train_warm_starter:
                self.warm_starter_delayed.load_state_dict(interpolate_state_dicts(self.warm_starter_delayed.state_dict(), self.warm_starter.state_dict(), self.ws_update_rate))

            # Run solver forward
            if self.use_residual_loss:
                Xs, primal_sols, residuals = self.solver(q, b, Pinv=Pinv, H=H, iters=self.qp_iter, return_residuals=True)
                primal_residual, dual_residual = residuals
                residual_loss = ((primal_residual ** 2).sum(dim=-1) + (dual_residual ** 2).sum(dim=-1)).mean()
                self.autonomous_losses["residual"] = 1e-3 * residual_loss
            else:
                Xs, primal_sols = self.solver(q, b, Pinv=Pinv, H=H, iters=self.qp_iter)
            sol = primal_sols[:, -1, :]

            # Compute warm starter loss
            if self.train_warm_starter:
                self.autonomous_losses["warm_starter"] = self.compute_warm_starter_loss(q, b, Pinv, H, Xs)

            # Compute imitation loss
            if self.imitate_mpc:
                # Use min(n of learned qp, n of mpc) as the common dimension of solution
                sol_dim = min(self.n_qp, mpc_sol.shape[-1])
                self.autonomous_losses["imitation_only"] = ((sol[:, :sol_dim] - mpc_sol[:, :sol_dim]) ** 2).sum(dim=-1).mean()

            if return_problem_params:
                problem_params = (torch.linalg.inv(Pinv), q, H, b)

        if not return_problem_params:
            # Only return the solution
            return sol
        else:
            # Return the solution as well as (P, q, H, b)
            return sol, problem_params
