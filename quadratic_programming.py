import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plotting import plot_quadratic_bezier_trajectory, plot_trajectory_in_3D

def quadratic_bezier(P0, P1, P2, t):
    """Compute a point on a quadratic Bézier curve."""
    return (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2

def bezier_derivative(P0, P1, P2, t):
    """Compute the first derivative of a quadratic Bézier curve."""
    return 2 * (1 - t) * (P1 - P0) + 2 * t * (P2 - P1)

def bezier_second_derivative(P0, P1, P2):
    """Compute the second derivative of a quadratic Bézier curve (constant)."""
    return 2 * (P2 - 2 * P1 + P0)

def interpolate_path_with_quadratic_bezier(path, total_time):
    """
    Generate a smooth trajectory using quadratic Bézier curves for each segment of the path.
    
    Args:
        path (list): A list of configurations from the RRT path.
        total_time (float): Total duration for the trajectory.
    
    Returns:
        A function that takes time t and returns the interpolated configuration, velocity, and acceleration.
    """
    # Extract configurations from the path
    configurations = [np.array(q[0]) for q in path]  # Assuming q[0] contains the position

    num_segments = len(configurations) - 1
    segment_duration = total_time / num_segments

    # Function to compute the trajectory
    def trajectory(t_real):
        # Determine which segment we're in
        segment_index = int(t_real // segment_duration)
        if segment_index >= num_segments:
            segment_index = num_segments - 1

        # Local time within the segment
        t_segment = (t_real % segment_duration) / segment_duration

        # Get the control points for this segment
        P0 = configurations[segment_index]
        P2 = configurations[segment_index + 1]
        P1 = (P0 + P2) / 2  # Simple choice: control point in the middle

        # Compute position, velocity, and acceleration
        q = quadratic_bezier(P0, P1, P2, t_segment)
        v = bezier_derivative(P0, P1, P2, t_segment) / segment_duration
        a = bezier_second_derivative(P0, P1, P2) / (segment_duration**2)

        return q, v, a

    return trajectory


# Main script
if __name__ == "__main__":
    # Example setup (assuming you have a way to get the path)
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from path import computepath

    robot, cube, viz = setupwithmeshcat()

    # Compute initial and goal configurations
    q = robot.q0.copy()
    q0, success_init = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz=None)
    qe, success_end = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz=None)

    if not (success_init and success_end):
        print("Error: Invalid initial or goal configuration")
        exit()

    # Compute the RRT path
    path, G = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, k=5000, delta_q=0.05)
    total_time = 5.0  # Total duration of 5 seconds

    # Create the quadratic Bézier trajectory
    trajectory = interpolate_path_with_quadratic_bezier(path, total_time)

    # plot path
    plot_trajectory_in_3D(path, G)

    # Plot the trajectory
    plot_quadratic_bezier_trajectory(trajectory, total_time, path)
