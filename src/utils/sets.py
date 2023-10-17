from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import numpy as np
from tqdm import tqdm_notebook as tqdm


def backward_reachable_set_linear(A_inv, B, X_set, x_min, x_max, u_min, u_max):
    """
    Compute the one-step backward reachable set for a linear system x_{k+1} = Ax + Bu.

    Parameters:
    A_inv (numpy.ndarray): Inverse of the A matrix in the system dynamics.
    B (numpy.ndarray): B matrix in the system dynamics.
    X_set (set): Set of points (as tuples) representing the current state space.
    x_min (float or numpy.ndarray): Minimum state constraints.
    x_max (float or numpy.ndarray): Maximum state constraints.
    u_min (float or numpy.ndarray): Minimum control input constraints.
    u_max (float or numpy.ndarray): Maximum control input constraints.

    Returns:
    set: One-step backward reachable set as a set of points (as tuples).
    """
    new_set = set()
    for x in X_set:
        for u in np.linspace(u_min, u_max, 5):
            prev_x = np.dot(A_inv, x - np.dot(B, u))
            if np.all(x_min <= prev_x) and np.all(prev_x <= x_max):
                new_set.add(tuple(prev_x))
    return new_set


def one_step_forward_reachable_set(g, S, x_min, x_max):
    """
    Compute the one-step forward reachable set for an autonomous system x_{k+1} = g(x_k).

    Parameters:
    g (function): Function representing the autonomous system dynamics.
    S (numpy.ndarray): Vertices of the initial set.
    x_min (numpy.ndarray): Minimum state constraints.
    x_max (numpy.ndarray): Maximum state constraints.

    Returns:
    numpy.ndarray: Vertices of the one-step forward reachable set.
    """
    new_vertices = []

    for x in S:
        next_x = g(x)

        # Check if the next state is within the state constraints
        if np.all(x_min <= next_x) and np.all(next_x <= x_max):
            new_vertices.append(next_x)

    return np.array(new_vertices)


def one_step_backward_reachable_set(g, S_hull, x_min, x_max, num_samples=1000):
    """
    Compute the one-step backward reachable set for an autonomous system x_{k+1} = g(x_k).

    Parameters:
    g (function): Function representing the autonomous system dynamics.
    S_hull (ConvexHull): Convex hull object of the initial set S.
    x_min (numpy.ndarray): Minimum state constraints.
    x_max (numpy.ndarray): Maximum state constraints.
    num_samples (int): Number of samples for approximation.

    Returns:
    numpy.ndarray: Vertices of the approximated one-step backward reachable set.
    """
    # Sample points within the state constraints
    sampled_points = np.random.uniform(x_min, x_max, (num_samples, len(x_min)))

    # Delaunay triangulation to speed up point-in-hull check
    delaunay_S = Delaunay(S_hull.points[S_hull.vertices, :])

    # Check which sampled points have their next state in S
    backward_reachable_points = []
    for x in sampled_points:
        next_x = g(x)
        if Delaunay.find_simplex(delaunay_S, next_x) >= 0:
            backward_reachable_points.append(x)

    # Compute the convex hull of the backward reachable points
    if len(backward_reachable_points) > 0:
        backward_hull = ConvexHull(np.array(backward_reachable_points))
        return backward_hull.points[backward_hull.vertices, :]
    else:
        return np.array([])


def compute_positive_invariant_set_from_origin(g, x_min, x_max, initial_radius=1.0, iterations=100):
    """
    Compute the positive invariant set for an autonomous system x_{k+1} = g(x_k) starting from a neighborhood of the origin.

    Parameters:
    g (function): Function representing the autonomous system dynamics.
    x_min (numpy.ndarray): Minimum state constraints.
    x_max (numpy.ndarray): Maximum state constraints.
    initial_radius (float): Radius of the initial neighborhood around the origin.
    iterations (int): Number of iterations for approximation.

    Returns:
    numpy.ndarray: Vertices of the approximated positive invariant set.
    """
    # Start from a neighborhood of the origin defined by the initial_radius
    initial_set = np.array([[initial_radius, 0], [0, initial_radius], [-initial_radius, 0], [0, -initial_radius]])
    current_set_hull = ConvexHull(initial_set)

    for _ in tqdm(range(iterations)):
        # Determine the sampling bounds based on the current set
        current_radius = np.max(np.linalg.norm(current_set_hull.points[current_set_hull.vertices, :], axis=1))
        sampling_min = np.maximum(x_min, -current_radius * 1.5)
        sampling_max = np.minimum(x_max, current_radius * 1.5)

        # Compute the one-step backward reachable set from the current set
        backward_reachable_vertices = one_step_backward_reachable_set(g, current_set_hull, sampling_min, sampling_max)

        # Update the current set to include the backward reachable set, effectively taking union
        if len(backward_reachable_vertices) > 0:
            new_hull = ConvexHull(np.vstack((current_set_hull.points[current_set_hull.vertices, :], backward_reachable_vertices)))
            current_set_hull = new_hull

    return current_set_hull.points[current_set_hull.vertices, :]


def compute_MCI(A, B, x_min, x_max, u_min, u_max, iterations=10):
    """
    Compute the Maximal Control Invariant (MCI) set for a given linear system x[k+1] = Ax[k] + Bu[k].

    Parameters:
    A (numpy.ndarray): State transition matrix.
    B (numpy.ndarray): Input matrix.
    x_min (numpy.ndarray): Minimum state constraints.
    x_max (numpy.ndarray): Maximum state constraints.
    u_min (numpy.ndarray): Minimum control input constraints.
    u_max (numpy.ndarray): Maximum control input constraints.
    iterations (int): Number of iterations for approximating the MCI set.

    Returns:
    numpy.ndarray: Vertices of the approximated MCI set.
    """

    # Precompute the inverse of A
    A_inv = np.linalg.inv(A)

    # Initialize the MCI set as a single point at the origin, using a set for uniqueness
    MCI_set = {(0, 0)}

    # Iteratively compute the MCI set
    for _ in range(iterations):
        MCI_set = backward_reachable_set_linear(A_inv, B, MCI_set, x_min, x_max, u_min, u_max)
        if len(MCI_set) == 0:
            break

    # Convert the set to an array for further processing or visualization
    MCI_array = np.array(list(MCI_set))

    if len(MCI_array) > 0:
        MCI_hull = ConvexHull(MCI_array)
        return MCI_hull.points[MCI_hull.vertices, :]
    else:
        return np.array([])