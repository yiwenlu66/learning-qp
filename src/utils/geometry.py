import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog


def find_interior_point(A, b):
    """
    Find an interior point of the polytope defined by Ax <= b using linear programming.
    
    Parameters:
    - A (numpy.ndarray): Coefficient matrix for inequalities.
    - b (numpy.ndarray): RHS vector for inequalities.
    
    Returns:
    - interior_point (numpy.ndarray): A point inside the polytope, or None if LP is infeasible.
    """
    num_vars = A.shape[1]
    
    # Objective function: zero coefficients as we only need a feasible solution
    c = np.zeros(num_vars)
    
    # Inequality constraints: Ax <= b
    eps = 1e-4
    A_ineq = A
    b_ineq = b - eps
    
    # Run linear programming to find a feasible point
    res = linprog(c, A_ub=A_ineq, b_ub=b_ineq, bounds=(None, None), method='highs')
    
    if res.success:
        from icecream import ic; ic(res.x)
        return res.x
    else:
        return None

