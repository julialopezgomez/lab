import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
import numpy as np

def plot_trajectory_in_3D(path, G, displayG=False):
    '''
    Creates a 3D plot of the trajectory, with lines connecting the points, and also plots all the vertices of the graph
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if(displayG):
        for _, _, c in G:
            ax.scatter(c.translation[0], c.translation[1], c.translation[2], c='grey', marker='o')


    if path is not None:
        for i in range(len(path)-1):
            q1, c1 = path[i]
            q2, c2 = path[i+1]
            ax.plot([c1.translation[0], c2.translation[0]], [c1.translation[1], c2.translation[1]], [c1.translation[2], c2.translation[2]], 'b')
   
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


# Example usage with 3D plotting
def plot_quadratic_bezier_trajectory(trajectory, total_time, original_path):
    """Plot both the quadratic Bézier trajectory and the original path configurations in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Sample the quadratic Bézier trajectory
    times = np.linspace(0, total_time, 100)
    x_traj, y_traj, z_traj = [], [], []
    for t in times:
        q, _, _ = trajectory(t)
        x_traj.append(q[0])
        y_traj.append(q[1])
        z_traj.append(q[2])

    # Plot the quadratic Bézier trajectory
    ax.plot(x_traj, y_traj, z_traj, 'r', label='Quadratic Bézier Trajectory')

    # Extract the original path coordinates
    x_orig, y_orig, z_orig = [], [], []
    for q, _ in original_path:  # Assuming each element in the path is (q, placement)
        x_orig.append(q[0])
        y_orig.append(q[1])
        z_orig.append(q[2])

    for i in range(len(original_path)-1):
        q1, c1 = original_path[i]
        q2, c2 = original_path[i+1]
        ax.plot([c1.translation[0], c2.translation[0]], [c1.translation[1], c2.translation[1]], [c1.translation[2], c2.translation[2]], 'b')

    # # Plot the original path
    # ax.scatter(x_orig, y_orig, z_orig, c='b', marker='o', label='Original Path', alpha=0.6)

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("3D Plot of Original Path and Quadratic Bézier Trajectory")

    plt.show()