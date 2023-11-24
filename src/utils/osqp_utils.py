import numpy as np
import qpsolvers

def osqp_solve_qp_guarantee_return(
    P, q, G=None, h=None, A=None, b=None, lb=None, ub=None, initvals=None, verbose=False, **kwargs,
):
    problem = qpsolvers.problem.Problem(P, q, G, h, A, b, lb, ub)
    solution = qpsolvers.solvers.osqp_.osqp_solve_problem(problem, initvals, verbose, **kwargs)
    sol_returned = solution.x if solution.x.dtype == np.float64 else np.zeros(q.shape[0])
    iter_count = solution.extras["info"].iter
    return sol_returned, iter_count

def osqp_oracle(q, b, P, H, return_iter_count=False, max_iter=1000):
    sol, iter_count = osqp_solve_qp_guarantee_return(
        P=P, q=q, G=-H, h=b,
        A=None, b=None, lb=None, ub=None,
        max_iter=max_iter, eps_abs=1e-10, eps_rel=1e-10,eps_prim_inf=1e-10, eps_dual_inf=1e-10, verbose=False,
    )
    if not return_iter_count:
        return sol
    else:
        return sol, iter_count
