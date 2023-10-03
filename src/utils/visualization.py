from .geometry import find_interior_point
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, HalfspaceIntersection


def plot_multiple_2d_polytopes_with_contour(polytope_contour_params):
    """
    Plot multiple 2D polytopes each defined by Ax <= b and overlay the contour of a quadratic function.
    
    Parameters:
    - polytope_contour_params (list of dict): List of dictionaries containing A, b, optimal_solution, P, q, and label.
    
    Returns:
    - fig (matplotlib.figure.Figure): Figure object.
    - ax (matplotlib.axes._subplots.AxesSubplot): Axis object.
    """
    
    fig, ax = plt.subplots()
    
    # Determine global x and y limits
    all_vertices = []
    for params in polytope_contour_params:
        interior_point = find_interior_point(params['A'], params['b'])
        if interior_point is not None:
            vertices = HalfspaceIntersection(np.hstack([params['A'], -params['b'][:, np.newaxis]]), interior_point).intersections
            all_vertices.append(vertices)
    all_vertices = np.vstack(all_vertices)
    
    margin = 0.5  # Additional margin around the polytopes
    x_range = np.max(all_vertices[:, 0]) - np.min(all_vertices[:, 0])
    y_range = np.max(all_vertices[:, 1]) - np.min(all_vertices[:, 1])
    max_range = max(x_range, y_range) + 2 * margin
    x_margin = (max_range - x_range) / 2
    y_margin = (max_range - y_range) / 2
    x_min, x_max = np.min(all_vertices[:, 0]) - x_margin, np.max(all_vertices[:, 0]) + x_margin
    y_min, y_max = np.min(all_vertices[:, 1]) - y_margin, np.max(all_vertices[:, 1]) + y_margin
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    custom_legend_handles = []
    
    for params in polytope_contour_params:
        A, b, P, q, color, label = params['A'], params['b'], params['P'], params['q'], params['color'], params['label']
        optimal_solution = params.get("optimal_solution", None)
        
        # Find an interior point
        interior_point = find_interior_point(A, b)
        if interior_point is None:
            continue  # Skip this polytope if LP is infeasible
        
        # Plot polytope
        halfspace_intersection = HalfspaceIntersection(np.hstack([A, -b[:, np.newaxis]]), interior_point)
        vertices = halfspace_intersection.intersections
        hull = ConvexHull(vertices)
        ordered_vertices = vertices[hull.vertices]
        closed_loop = np.vstack([ordered_vertices, ordered_vertices[0]])
        
        ax.fill(closed_loop[:, 0], closed_loop[:, 1], alpha=0.3, color=color, label=f"{label} (Polytope)")
        ax.plot(closed_loop[:, 0], closed_loop[:, 1], color=color)
        
        # Mark the optimal solution
        if optimal_solution is not None:
            ax.plot(optimal_solution[0], optimal_solution[1], 'o', color=color)
                
        # Evaluate quadratic function
        Z = np.zeros_like(x_grid)
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                x_vec = np.array([x_grid[i, j], y_grid[i, j]])
                Z[i, j] = 0.5 * x_vec.T @ P @ x_vec + q.T @ x_vec
                
        # Plot contour
        contour = ax.contour(x_grid, y_grid, Z, levels=5, colors=color)  # Reduced number of levels for sparser contour

        # Create a custom legend handle
        custom_legend_handles.append(Line2D([0], [0], color=color, lw=4, label=label))

    # Adjust plot settings
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Add custom legend
    if custom_legend_handles:
        # Move legend outside the plot
        ax.legend(handles=custom_legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
        # Adjust layout to prevent clipping
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    return fig, ax
