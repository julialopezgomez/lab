import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tools import setupwithmeshcat
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from inverse_geometry import computeqgrasppose
from path import computepath

# Quadratic Bézier functions
def quadratic_bezier(P0, P1, P2, t):
    """Compute a point on a quadratic Bézier curve."""
    return (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2

def compute_smooth_control_points(cube_placements):
    """
    Compute smooth control points for each segment of the path using quadratic programming.
    
    Args:
        cube_placements (list): A list of SE3 cube placements from the RRT path.
    
    Returns:
        A list of optimized control points for the Bézier curves.
    """
    n = len(cube_placements) - 1  # Number of segments
    dim = 3  # Dimension (x, y, z)
    
    # Variables: Control points for each segment (P1)
    P1 = [cp.Variable(dim) for _ in range(n)]
    
    # Constraints
    constraints = []
    
    # Objective: Minimize the sum of squared differences between consecutive control points
    # to ensure smoothness
    objective_terms = []
    for i in range(n):
        # Position constraints at t=0 and t=1
        P0 = cube_placements[i].translation
        P2 = cube_placements[i + 1].translation

        # No need to add constraints for P0 and P2 since they are fixed

        if i < n - 1:
            # Velocity continuity between segments
            # Compute derivative at t=1 for segment i and derivative at t=0 for segment i+1
            # B'_i(1) = 2(P2 - P1_i)
            # B'_{i+1}(0) = 2(P1_{i+1} - P0_{i+1})
            constraints.append(2 * (P2 - P1[i]) == 2 * (P1[i + 1] - cube_placements[i + 1].translation))

        if i > 0:
            # Optional: Minimize the change in acceleration (smoothness)
            acc_i = 2 * (P2 - 2 * P1[i] + P0)
            P0_prev = cube_placements[i - 1].translation
            P2_prev = P0
            acc_prev = 2 * (P2_prev - 2 * P1[i - 1] + P0_prev)
            objective_terms.append(cp.norm(acc_i - acc_prev, 2)**2)
        else:
            # For the first segment, we can minimize the acceleration
            P0 = cube_placements[i].translation
            P2 = cube_placements[i + 1].translation
            acc_i = 2 * (P2 - 2 * P1[i] + P0)
            objective_terms.append(cp.norm(acc_i, 2)**2)

    # Sum the objective terms
    objective = cp.Minimize(cp.sum(objective_terms))

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Retrieve optimized control points
    control_points = [P1_i.value for P1_i in P1]

    return control_points

def interpolate_cube_path_with_quadratic_bezier(cube_placements, total_time):
    """
    Generate a smooth trajectory for the cube using quadratic Bézier curves for each segment.
    
    Args:
        cube_placements (list): A list of SE3 cube placements from the RRT path.
        total_time (float): Total duration for the trajectory.
    
    Returns:
        A function that takes time t and returns the smooth cube placement.
    """
    num_segments = len(cube_placements) - 1
    segment_duration = total_time / num_segments

    # Compute smooth control points
    control_points = compute_smooth_control_points(cube_placements)

    def cube_trajectory(t_real):
        # Determine which segment we're in
        segment_index = int(t_real // segment_duration)
        if segment_index >= num_segments:
            segment_index = num_segments - 1

        # Local time within the segment
        t_segment = (t_real % segment_duration) / segment_duration

        # Get the control points for this segment
        P0 = cube_placements[segment_index].translation
        P2 = cube_placements[segment_index + 1].translation
        P1 = control_points[segment_index]  # Use the smooth control point

        # Compute the position of the cube
        position = quadratic_bezier(P0, P1, P2, t_segment)

        # Construct a new SE3 placement for the cube
        placement = cube_placements[segment_index].copy()
        placement.translation = position

        return placement

    return cube_trajectory

def compute_robot_trajectory_from_cube_trajectory(robot, cube, cube_trajectory, total_time, num_samples=100):
    """
    Compute the robot configurations for the smooth cube trajectory using inverse kinematics.
    
    Args:
        robot: The robot model.
        cube: The cube object.
        cube_trajectory: The smooth cube trajectory function.
        total_time: Total duration of the trajectory.
        num_samples: Number of samples to evaluate the trajectory.
    
    Returns:
        A list of robot configurations corresponding to the smooth cube trajectory.
    """
    robot_configurations = []

    for t in np.linspace(0, total_time, num_samples):
        cube_placement = cube_trajectory(t)
        q, success = computeqgrasppose(robot, robot.q0.copy(), cube, cube_placement)
        if not success:
            raise ValueError("Failed to compute a valid robot configuration for the given cube placement.")
        robot_configurations.append(q)

    return robot_configurations

def plot_cube_trajectory(cube_placements, smooth_trajectory, total_time):
    """
    Plot the original and smooth cube trajectories in 3D.
    
    Args:
        cube_placements: Original cube placements from the RRT path.
        smooth_trajectory: The smooth cube trajectory function.
        total_time: Total duration of the trajectory.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original cube placements
    x_orig, y_orig, z_orig = [], [], []
    for placement in cube_placements:
        x_orig.append(placement.translation[0])
        y_orig.append(placement.translation[1])
        z_orig.append(placement.translation[2])
    ax.plot(x_orig, y_orig, z_orig, 'bo--', label='Original Cube Path', alpha=0.5)

    # Plot the smooth cube trajectory
    x_smooth, y_smooth, z_smooth = [], [], []
    for t in np.linspace(0, total_time, 100):
        placement = smooth_trajectory(t)
        x_smooth.append(placement.translation[0])
        y_smooth.append(placement.translation[1])
        z_smooth.append(placement.translation[2])
    ax.plot(x_smooth, y_smooth, z_smooth, 'r-', label='Smooth Cube Trajectory')

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("3D Plot of Original and Smooth Cube Trajectories")

    plt.show()

# Main script
if __name__ == "__main__":
    # Setup the robot, cube, and visualization
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
    
    # Extract cube placements from the path
    cube_placements = [c for (q, c) in path]
    total_time = 5.0  # Total duration of 5 seconds

    # Create the smooth cube trajectory
    smooth_cube_trajectory = interpolate_cube_path_with_quadratic_bezier(cube_placements, total_time)

    # Compute the robot configurations for the smooth cube trajectory
    try:
        robot_trajectory = compute_robot_trajectory_from_cube_trajectory(robot, cube, smooth_cube_trajectory, total_time)
    except ValueError as e:
        print(str(e))
        exit()

    # Plot the original and smooth cube trajectories
    plot_cube_trajectory(cube_placements, smooth_cube_trajectory, total_time)
