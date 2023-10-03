import numpy as np
import qpsolvers

def osqp_solve_qp_guarantee_return(
    P, q, G=None, h=None, A=None, b=None, lb=None, ub=None, initvals=None, verbose=False, **kwargs,
):
    problem = qpsolvers.problem.Problem(P, q, G, h, A, b, lb, ub)
    solution = qpsolvers.solvers.osqp_.osqp_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.x.dtype == np.float64 else np.zeros(q.shape[0])

def osqp_oracle(q, b, P, H):
    return osqp_solve_qp_guarantee_return(
        P=P, q=q, G=-H, h=b,
        A=None, b=None, lb=None, ub=None,
        max_iter=30000, eps_abs=1e-10, eps_rel=1e-10,eps_prim_inf=1e-10, eps_dual_inf=1e-10, verbose=False
    )

