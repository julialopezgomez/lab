#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau and s2107370
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, norm
from pinocchio.utils import rotate

from config import LEFT_HAND, RIGHT_HAND, EPSILON
from tools import collision, setcubeplacement, getcubeplacement
import time

from tools import setupwithmeshcat
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from inverse_geometry import computeqgrasppose

import matplotlib.pyplot as plt
from plotting import plot_trajectory_in_3D

REACHED, ADVANCED, TRAPPED = "REACHED", "ADVANCED", "TRAPPED"

# set the limits for the random cube placement
x_min, x_max = 0.3, 0.42  # X-axis limits could be 0.33, 0.4
y_min, y_max = -0.4, 0.14  # Y-axis limits could be -0.3, 0.11
z_min, z_max = 0.93, 1.3  # Z-axis limits 

def generate_random_cube_placement():
    '''
    Generate a random cube placement within the constraints, and check for collisions.
    '''
    # sample a random cube placement uniformly within the constraints
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)

    placement = pin.SE3(rotate('z', 0), np.array([x, y, z]))

    return placement

def distance(c1,c2):
    '''
    Returns the distance between two cube placements, 
    without considering the orientation.
    '''
    return norm(c1.translation-c2.translation)

def nearest_vertex(G, c_rand):
    '''
    Finds the nearest vertex index in the graph G to the random configuration c_rand.
    Returns the index of the nearest vertex
    '''
    min_dist = np.inf
    nearest_vertex = None
    for (i,(_, _, c)) in enumerate(G):
        dist = distance(c, c_rand)
        if dist < min_dist:
            min_dist = dist
            nearest_vertex = i
    return nearest_vertex

def add_edge_and_vertex(G, parent, q, c):
    '''
    Add a new vertex to the graph, including parent, configuration and cube placement.
    '''
    G += [(parent, q, c)]

def lerp(p0, p1, t):
    '''
    Linear interpolation between two points p0 and p1.
    '''
    return p0*(1-t) + p1*t

def lerp_cube(cube_0, cube_1, t):
    '''
    Linear interpolation between two cube placements cube_0 and cube_1. 
    Returns a new cube placement.
    '''
    new_placement = lerp(cube_0.translation, cube_1.translation, t)
    return pin.SE3(rotate('z', 0), new_placement)

def new_placement(robot, cube, q_near, c_near, c_rand, discretisationsteps, delta_q):
    '''
    Compute a new configuration and cube placement by interpolating between c_near and c_rand.
    The maximum distance between c_near and c_rand is set by delta_q.

    Return:
    - q_end: the new configuration
    - c_end: the new cube placement
    - outcome: REACHED if the interpolation reached c_rand, ADVANCED if the interpolation was limited by delta_q, TRAPPED if a collision was detected.
    '''
    c_end = c_rand.copy()
    dist = distance(c_near, c_rand)
    outcome = REACHED
    if delta_q is not None and dist > delta_q:
        # interpolate between c_near and c_rand with a maximum distance of delta_q
        c_end = lerp_cube(c_near, c_rand, delta_q/dist)
        outcome = ADVANCED

    dt = 1 / discretisationsteps # time step for the interpolation

    q_prev = q_near.copy()
    c_prev = c_near.copy()

    for i in range(1,discretisationsteps + 1):
        c = lerp_cube(c_near, c_end, i*dt) # interpolate between c_near and c_end
        q_end, valid = computeqgrasppose(robot, q_prev, cube, c) # compute the new configuration

        if not valid:
            # if a collision is detected in the new configuration, return the previous configuration and cube placement
            outcome = TRAPPED
            return q_prev, c_prev, outcome
        
        # update the previous configuration and cube placement
        q_prev = q_end
        c_prev = c

    return q_end, c_end, outcome


def extend(robot, cube, G, c_rand, discretisationsteps, delta_q):
    '''
    Finds the configuration q_new and cube placement c_new by extending the nearest vertex of 
    graph G towards c_rand. q_new and c_new are added to the graph G.

    Return:
    - q_new: the new configuration
    - c_new: the new cube placement
    - outcome: REACHED if the interpolation reached c_rand, ADVANCED if the interpolation was limited by delta_q, TRAPPED if a collision was detected.
    '''
    # find the nearest vertex in G to c_rand
    c_near_idx = nearest_vertex(G, c_rand)
    q_near, c_near = G[c_near_idx][1], G[c_near_idx][2]

    # interpolate between c_near and c_rand
    q_new, c_new, outcome = new_placement(robot, cube, q_near, c_near, c_rand, discretisationsteps, delta_q)

    # add the new configuration and cube placement to the graph
    add_edge_and_vertex(G, c_near_idx, q_new, c_new)
        
    return q_new, c_new, outcome


def connect(robot, cube, G, c, discretisationsteps, delta_q):
    '''
    Connects a new configuration c_new to the graph G by finding the nearest vertex and adding an edge.
    Returns True if the connection was successful, False otherwise.
    '''
    while True:
        # extend the graph towards cube placement c
        _, _, outcome = extend(robot, cube, G, c, discretisationsteps, delta_q)
        if outcome != ADVANCED:
            # break if the interpolation has reached c or a collision was detected
            break
    
    # return True if the interpolation reached c, False otherwise
    return outcome == REACHED

def swap(G1, G2):
    '''
    Swap the connected graphs G1 and G2.
    '''
    return G2.copy(), G1.copy()


def RRT_CONNECT(robot, cube, q_init, q_goal, cubeplacementq0, cubeplacementqgoal, k, delta_q):
    '''
    RRT-Connect algorithm to find a path between q_init and q_goal.
    Returns the connected graphs G_start and G_end, and a boolean indicating if a path was found.
    '''

    discretisationsteps_newconf = 200 # number of steps for the interpolation

    # initialise initial and goal configurations and cube placements
    q_init = q_init.copy()
    q_goal = q_goal.copy()
    c_init = cubeplacementq0
    c_goal = cubeplacementqgoal

    # initialise the connected graphs G_start and G_end
    G_start = [(None, q_init, c_init)]
    G_end = [(None, q_goal, c_goal)]

    G1, G2 = G_start, G_end

    for i in range(k):
        print("Iteration", i)

        c_rand = generate_random_cube_placement() # generate a random cube placement

        # extend the graph G1 towards c_rand
        _, c_new, _ = extend(robot, cube, G1, c_rand, discretisationsteps_newconf, delta_q)

        # if the new configuration is connected to G2, a path is found
        if connect(robot, cube, G2, c_new, discretisationsteps_newconf, delta_q):
            print("Path found")
            if i%2 == 0:
                return G1, G2, True
            else:
                return G2, G1, True
        
        # swap the connected graphs
        G1, G2 = swap(G1, G2)
        
    print("Path not found")

    return G1, G2, False


def get_path(G, with_cube=False):
    '''
    Compute the path from the last vertex of the graph G to the root.
    If with_cube is True, the path is returned as a list of (q, c) pairs, 
    otherwise only the configurations q are returned.
    '''
    path = []
    node = G[-1]

    if with_cube:
        while node[0] is not None:
            path = [(node[1], node[2])] + path
            node = G[node[0]]
        path = [(G[0][1], G[0][2])] + path

    else:
        while node[0] is not None:
            path = [node[1]] + path
            node = G[node[0]]
        path = [G[0][1]] + path

    return path

def plot_connected_graphs(G1, G2):
    '''
    Plot the connected graphs G1 and G2 in 3D, each with a different color, and lines connecting the connected vertices.
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for _, _, c in G1:
        ax.scatter(c.translation[0], c.translation[1], c.translation[2], c='blue', marker='o')

    for _, _, c in G2:
        ax.scatter(c.translation[0], c.translation[1], c.translation[2], c='red', marker='o')

    path1 = get_path(G1, with_cube=True)
    path2 = get_path(G2, with_cube=True)

    for i in range(len(path1)-1):
        _, c1 = path1[i]
        _, c2 = path1[i+1]
        ax.plot([c1.translation[0], c2.translation[0]], 
                [c1.translation[1], c2.translation[1]], 
                [c1.translation[2], c2.translation[2]], 'b')

    for i in range(len(path2)-1):
        _, c1 = path2[i]
        _, c2 = path2[i+1]
        ax.plot([c1.translation[0], c2.translation[0]], 
                [c1.translation[1], c2.translation[1]], 
                [c1.translation[2], c2.translation[2]], 'r')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set the title of the plot to the index of the iteration
    ax.set_title(f'Graph coverage and paths generated')

    # set the legend of the plot    
    ax.legend(['Path 1', 'Path 2'])

    # set the limits of the plot so that it is always in the same scale
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    # save image in img directory with name of index i
    fig.savefig(f'img/path_plot.png')

    plt.close(fig)



def computepath(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, k=5000, delta_q=0.05):
    '''
    Compute a collision-free path from qinit to qgoal under grasping constraints.
    The path is expressed as a list of configurations.
    '''
    
    # run the RRT-Connect algorithm to find a path between qinit and qgoal
    G1, G2, pathfound = RRT_CONNECT(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, k, delta_q)

    # plot the connected graphs G1 and G2
    plot_connected_graphs(G1, G2)
    
    if not pathfound:
        return None
    
    path1 = get_path(G1)
    path2 = get_path(G2)
    
    return path1 + path2[::-1]


def displaypath(robot,path,dt,viz):
    '''
    Display the path in the meshcat visualizer.
    '''
    if path is None:
        return
    
    # for q, c in path:
    #     setcubeplacement(robot, cube, c)
    #     viz.display(q)
    #     time.sleep(dt)

    setcubeplacement(robot, cube, CUBE_PLACEMENT)
    for q in path:
        viz.display(q)
        time.sleep(dt)

if __name__ == "__main__":

    robot, cube, viz = setupwithmeshcat()
    

    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz=None)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz=None)
    
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    if path is not None:
        # Hey Markers! This is implemented for you to visualize the path in Meshcat
        # Hit Enter to display the path, and q to quit.
        input("Press Enter to display the path")
        
        while True:
            displaypath(robot,path,dt=0.05,viz=viz) # you ll probably want to lower dt
            if input("Press Enter to display the path again, type 'q' to quit: ") == 'q':
                break