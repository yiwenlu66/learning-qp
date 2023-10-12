import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from itertools import combinations


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


def find_supporting_hyperplanes(vertices_2D):
    """
    Given a set of 2D vertices, find the supporting hyperplanes of the convex hull.
    
    Parameters:
    - vertices_2D (numpy.ndarray): 2D vertices of the polytope.
    
    Returns:
    - A_2D (numpy.ndarray): The coefficient matrix for the 2D inequalities.
    - b_2D (numpy.ndarray): The constant terms for the 2D inequalities.
    """
    A_list = []
    b_list = []
    hull = ConvexHull(vertices_2D)
    centroid = np.mean(vertices_2D, axis=0)
    
    for simplex in hull.simplices:
        v1, v2 = vertices_2D[simplex]
        edge = v2 - v1
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        
        # Choose the direction of the normal so that it points away from the centroid of the polytope
        if np.dot(normal, centroid - v1) > 0:
            normal = -normal
        
        b = np.dot(normal, v1)
        A_list.append(normal)
        b_list.append(b)
        
    return np.array(A_list), np.array(b_list)



def high_dim_to_2D(A, b):
    """
    Converts a high-dimensional polytope {x | Ax <= b} to its 2D projection {x | A_proj x <= b_proj}.
    
    Parameters:
    - A (numpy.ndarray): The coefficient matrix for the high-dimensional inequalities.
    - b (numpy.ndarray): The constant terms for the high-dimensional inequalities.
    
    Returns:
    - A_2D (numpy.ndarray): The coefficient matrix for the 2D inequalities.
    - b_2D (numpy.ndarray): The constant terms for the 2D inequalities.
    """
    def find_high_dim_vertices(A, b):
        n = A.shape[1]
        m = A.shape[0]
        vertices = []
        for idx in combinations(range(m), n):
            A_sub = A[idx, :]
            b_sub = b[list(idx)]
            if np.linalg.matrix_rank(A_sub) == n:
                try:
                    x = np.linalg.solve(A_sub, b_sub)
                except np.linalg.LinAlgError:
                    continue
                if all(np.dot(A, x) <= b + 1e-9):
                    vertices.append(x)
        return np.array(vertices)
    
    # Step 1: Find high-dimensional vertices
    vertices_high_dim = find_high_dim_vertices(A, b)
    
    # Step 2: Project to 2D
    vertices_2D = vertices_high_dim[:, :2]
    
    # Step 3: Find supporting hyperplanes in 2D
    A_2D, b_2D = find_supporting_hyperplanes(vertices_2D)
    
    return A_2D, b_2D


def high_dim_to_2D_sampling(A, b, grid_size=50, x_range=(-1, 1)):
    """
    Converts a high-dimensional polytope {x | Ax <= b} to its 2D projection {x | A_proj x <= b_proj}
    using a sampling-based approximation method.
    
    Parameters:
    - A (numpy.ndarray): The coefficient matrix for the high-dimensional inequalities.
    - b (numpy.ndarray): The constant terms for the high-dimensional inequalities.
    - grid_size (int): The number of grid points along each dimension in the sampling grid.
    - x_range (tuple): The range (min, max) for both x1 and x2 in the 2D plane.
    
    Returns:
    - A_2D (numpy.ndarray): The coefficient matrix for the 2D inequalities.
    - b_2D (numpy.ndarray): The constant terms for the 2D inequalities.
    """
    
    def sample_based_projection_LP(A, b, x1_range, x2_range, grid_size):
        x1_min, x1_max = x1_range
        x2_min, x2_max = x2_range
        x1_vals = np.linspace(x1_min, x1_max, grid_size)
        x2_vals = np.linspace(x2_min, x2_max, grid_size)
        grid_points = np.array([[x1, x2] for x1 in x1_vals for x2 in x2_vals])
        feasible_points = []
        for point in grid_points:
            x_dim = np.zeros(A.shape[1])
            x_dim[:2] = point
            c = np.zeros(A.shape[1] - 2)
            A_ub = A[:, 2:]
            b_ub = b - np.dot(A[:, :2], point)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')
            if res.success:
                feasible_points.append(point)
        feasible_points = np.array(feasible_points)
        if feasible_points.shape[0] < 3:
            return "Insufficient feasible points for a 2D polytope."
        hull = ConvexHull(feasible_points)
        vertices = hull.points[hull.vertices]
        return vertices
    
    # Step 1: Sample points and find the approximated vertices in 2D
    vertices_approx = sample_based_projection_LP(A, b, x_range, x_range, grid_size)
    
    # Step 2: Find supporting hyperplanes in 2D
    A_2D, b_2D = find_supporting_hyperplanes(vertices_approx)
    
    return A_2D, b_2D


def partial_minimization_2D(P, q):
    """
    Performs partial minimization over dimensions starting from 3 to obtain a 2D quadratic function.
    
    Parameters:
    - P (numpy.ndarray): The coefficient matrix for the high-dimensional quadratic function.
    - q (numpy.ndarray): The coefficient vector for the high-dimensional quadratic function.
    
    Returns:
    - P_2D (numpy.ndarray): The 2x2 coefficient matrix for the resulting 2D quadratic function.
    - q_2D (numpy.ndarray): The 2D coefficient vector for the resulting 2D quadratic function.
    - c (float): The constant bias term for the resulting 2D quadratic function.
    """
    # Decompose P into P11, P12, P21, P22
    P11 = P[:2, :2]
    P12 = P[:2, 2:]
    P21 = P[2:, :2]
    P22 = P[2:, 2:]
    
    # Decompose q into q1 and q2
    q1 = q[:2]
    q2 = q[2:]

    # Compute the 2D quadratic function parameters
    P_2D = P11 - P12 @ np.linalg.inv(P22) @ P21
    q_2D = q1 - P12 @ np.linalg.inv(P22) @ q2
    c = -0.5 * q2 @ np.linalg.inv(P22) @ q2

    return P_2D, q_2D, c
