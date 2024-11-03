import numpy as np
from bezier import Bezier

from path import computepath
from tools import setupwithmeshcat
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from inverse_geometry import computeqgrasppose
    

def interpolate_path(path, total_time):
    """
    Create a time-parameterized trajectory for the given path.
    
    Args:
        path (list): A list of configurations [(q1, c1), (q2, c2), ...] along the path.
        total_time (float): The total duration of the trajectory.

    Returns:
        A function that takes time t and returns the configuration, velocity, and acceleration.
    """
    # Extract configurations from the path
    configurations = [q for (q, c) in path]
    
    # Ensure that we have at least two configurations to create a Bezier curve
    if len(configurations) < 2:
        raise ValueError("Path must have at least two configurations to create a Bezier curve.")
    
    # Create a Bezier curve for smooth interpolation
    bezier_curve = Bezier(configurations, t_max=total_time)

    # Compute the first derivative (velocity)
    if bezier_curve.size_ > 0:
        velocity_curve = bezier_curve.derivative(1)
    else:
        velocity_curve = lambda t: np.zeros(bezier_curve.dim_)

    # Compute the second derivative (acceleration)
    if velocity_curve.size_ > 0:
        acceleration_curve = velocity_curve.derivative(1)
    else:
        acceleration_curve = lambda t: np.zeros(bezier_curve.dim_)

    def trajectory(t):
        if t < 0:
            t = 0
        elif t > total_time:
            t = total_time
        
        # Evaluate configuration, velocity, and acceleration at time t
        q = bezier_curve(t)
        v = velocity_curve(t) if callable(velocity_curve) else np.zeros(bezier_curve.dim_)
        a = acceleration_curve(t) if callable(acceleration_curve) else np.zeros(bezier_curve.dim_)
        return q, v, a

    return trajectory

if __name__ == "__main__":
    robot, cube, viz = setupwithmeshcat()
    # Example usage:
    q = robot.q0.copy()
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz=None)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz=None)

    # Pass `robot` and `cube` to `computepath`
    path, G = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, k=5000, delta_q=0.05)

    total_time = 5.0  # Total duration of 5 seconds
    traj = interpolate_path(path, total_time)

    # Test the trajectory at different time instances
    for t in np.linspace(0, total_time, 10):
        q, v, a = traj(t)
        print(f"t={t:.2f}: q={q}, v={v}, a={a}")