import pinocchio as pin
import numpy as np
from numpy.linalg import norm
from pinocchio.utils import rotate

from config import LEFT_HAND, RIGHT_HAND, EPSILON
from tools import collision, setcubeplacement, getcubeplacement
import time

from tools import setupwithmeshcat
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from inverse_geometry import computeqgrasppose

import matplotlib.pyplot as plt
from plotting import plot_trajectory_in_3D

def generate_random_cube_placement(robot, cube, q_current):
    """Generate a random cube placement within constraints."""
    x_min, x_max = 0.33, 0.4
    y_min, y_max = -0.3, 0.11
    z_min, z_max = 0.93, 1.1

    counter = 0
    while True:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)

        placement = pin.SE3(rotate('z', 0), np.array([x, y, z]))
        setcubeplacement(robot, cube, placement)
        q_rand = q_current.copy()
        q_rand, not_in_collision = computeqgrasppose(robot, q_rand, cube, placement)

        if not_in_collision:
            return q_rand, placement
        
        counter += 1
        if counter > 100:
            break

    print("Error: Could not find a valid random configuration")
    return None

def distance(c1, c2):
    """Returns the Euclidean distance between two cube placements."""
    return norm(c1.translation - c2.translation)

def nearest_vertex(G, c_rand):
    """Finds the nearest vertex in the graph to a random configuration."""
    min_dist = np.inf
    nearest_vertex = None
    for i, (_, _, c, _) in enumerate(G):
        dist = distance(c, c_rand)
        if dist < min_dist:
            min_dist = dist
            nearest_vertex = i
    return nearest_vertex

def add_edge_and_vertex(G, parent, q, c, cost):
    """Adds a new vertex and edge to the graph."""
    G.append((parent, q, c, cost))

def lerp(q0, q1, t):
    """Linear interpolation between two configurations."""
    return q0 * (1 - t) + q1 * t

def lerp_cube(cube_0, cube_1, t):
    """Linear interpolation between two cube placements."""
    new_placement = lerp(cube_0.translation, cube_1.translation, t)
    return pin.SE3(rotate('z', 0), new_placement)

def new_placement(robot, cube, q_near, c_near, c_rand, discretisationsteps, delta_q=None):
    """Compute new placement along the direction of c_rand within delta_q limits."""
    c_end = c_rand.copy()
    dist = distance(c_near, c_rand)
    if delta_q is not None and dist > delta_q:
        c_end = lerp_cube(c_near, c_rand, delta_q / dist)

    dt = 1 / discretisationsteps
    q_prev = q_near.copy()
    c_prev = c_near.copy()
    for i in range(1, discretisationsteps + 1):
        c = lerp_cube(c_near, c_end, i * dt)
        q_end, valid = computeqgrasppose(robot, q_prev, cube, c)
        if not valid:
            return q_prev, c_prev
        q_prev = q_end
        c_prev = c

    return q_end, c

def neighborhood(G, c_rand, radius):
    """Find nodes within a given radius of c_rand in the graph."""
    neighbors = []
    for i, (_, _, c, cost) in enumerate(G):
        if distance(c, c_rand) < radius:
            neighbors.append((i, cost))
    return neighbors

def rewire(G, new_idx, neighbors, robot, cube, discretisationsteps):
    """Rewire nodes in the graph to ensure optimal paths."""
    q_new, c_new, cost_new = G[new_idx][1:4]
    for neighbor_idx, neighbor_cost in neighbors:
        q_neighbor, c_neighbor, cost_neighbor = G[neighbor_idx][1:4]
        potential_cost = cost_new + distance(c_new, c_neighbor)

        if potential_cost < cost_neighbor:
            q_intermediate, c_intermediate = new_placement(robot, cube, q_new, c_new, c_neighbor, discretisationsteps)
            if np.allclose(c_intermediate.translation, c_neighbor.translation):
                G[neighbor_idx] = (new_idx, q_neighbor, c_neighbor, potential_cost)

def valid_edge_to_goal(robot, cube, q_new, c_new, c_goal, discretisationsteps, delta_q=0.01):
    """Check if there exists a valid edge from c_new to c_goal."""
    return norm(c_goal.translation - new_placement(robot, cube, q_new, c_new, c_goal, discretisationsteps, delta_q)[1].translation) < delta_q

def RRT_star(robot, cube, q_init, q_goal, k=1000, delta_q=0.01, cubeplacementq0=None, cubeplacementqgoal=None):
    """RRT* algorithm for optimized path generation."""
    discretisationsteps_newconf = 200
    discretisationsteps_validedge = 200

    q_init = q_init.copy()
    q_goal = q_goal.copy()
    c_init = cubeplacementq0
    c_goal = cubeplacementqgoal
    G = [(None, q_init, c_init, 0)]

    for i in range(k):
        print("Iteration", i)
        _, c_rand = generate_random_cube_placement(robot, cube, q_init)
        c_near_idx = nearest_vertex(G, c_rand)
        q_near, c_near, cost_near = G[c_near_idx][1:4]
        q_new, c_new = new_placement(robot, cube, q_near, c_near, c_rand, discretisationsteps_newconf, delta_q)

        radius = 0.1
        neighbors = neighborhood(G, c_new, radius)

        min_cost = cost_near + distance(c_near, c_new)
        best_parent = c_near_idx
        for neighbor_idx, neighbor_cost in neighbors:
            q_neighbor, c_neighbor, cost_neighbor = G[neighbor_idx][1:4]
            edge_cost = distance(c_neighbor, c_new)
            if cost_neighbor + edge_cost < min_cost:
                q_intermediate, c_intermediate = new_placement(robot, cube, q_neighbor, c_neighbor, c_new, discretisationsteps_newconf)
                if np.allclose(c_intermediate.translation, c_new.translation):
                    min_cost = cost_neighbor + edge_cost
                    best_parent = neighbor_idx

        add_edge_and_vertex(G, best_parent, q_new, c_new, min_cost)
        new_idx = len(G) - 1
        rewire(G, new_idx, neighbors, robot, cube, discretisationsteps_validedge)

        if valid_edge_to_goal(robot, cube, q_new, c_new, c_goal, discretisationsteps_validedge, delta_q):
            print("Path found")
            add_edge_and_vertex(G, new_idx, q_goal, c_goal, min_cost + distance(c_new, c_goal))
            return G, True

    print("Path not found")
    return G, False

def get_path(G):
    """Retrieve the optimal path from the graph G."""
    path = []
    node = G[-1]
    while node[0] is not None:
        path = [(node[1], node[2])] + path
        node = G[node[0]]
    path = [(G[0][1], G[0][2])] + path
    return path

def computepath(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, k=5000, delta_q=0.01):
    """Compute an optimized RRT* path from start to goal."""
    G, pathfound = RRT_star(robot, cube, qinit, qgoal, k, delta_q, cubeplacementq0, cubeplacementqgoal)
    if not pathfound:
        return None, G
    return get_path(G), G

def displaypath(robot, path, dt, viz):
    """Display the path in the visualization."""
    if path is None:
        return
    for q, c in path:
        setcubeplacement(robot, cube, c)
        viz.display(q)
        time.sleep(dt)

if __name__ == "__main__":
    robot, cube, viz = setupwithmeshcat()
    q = robot.q0.copy()
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz=None)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz=None)

    if not (successinit and successend):
        print("Error: Invalid initial or goal configuration")
        exit()

    path, G = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, k=5000, delta_q=0.05)

    input("Press Enter to display the path")
    while True:
        displaypath(robot, path, dt=0.05, viz=viz)
        if input("Press Enter to display the path again, type 'q' to quit") == 'q':
            break

    input("Press Enter to plot the path in 3D")
    plot_trajectory_in_3D(path, G, displayG=True)


# comparisons, reports (benchmarking)
# change varialbes in the code (discuss the results) - change something in the RRT* algorithm, or compare it with a different method