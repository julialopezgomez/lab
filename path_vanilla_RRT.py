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

# set the limits for the random cube placement
x_min, x_max = 0.3, 0.42  # X-axis limits could be 0.33, 0.4
y_min, y_max = -0.4, 0.14  # Y-axis limits could be -0.3, 0.11
z_min, z_max = 0.93, 1.3  # Z-axis limits 

def generate_random_cube_placement():
    '''
    Generate a random cube placement within the constraints, and check for collisions.
    '''
    # Generate a random cube placement within the constraints
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)

    placement = pin.SE3(rotate('z', 0), np.array([x, y, z]))
    
    setcubeplacement(robot, cube, placement)

    return placement 


def distance(c1,c2):
    '''
    Returns the distance between two cube placements, 
    without considering the orientation
    '''
    return norm(c1.translation-c2.translation)

def nearest_vertex(G, c_rand):
    '''
    Finds the nearest vertex in the graph G to the random configuration c_rand
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
    '''
    c_end = c_rand.copy()
    dist = distance(c_near, c_rand)
    if delta_q is not None and dist > delta_q:
        # interpolate between c_near and c_rand with a maximum distance of delta_q
        c_end = lerp_cube(c_near, c_rand, delta_q/dist)

    dt = 1 / discretisationsteps # time step for the interpolation

    q_prev = q_near.copy()
    c_prev = c_near.copy()
    
    for i in range(1,discretisationsteps + 1):
        c = lerp_cube(c_near, c_end, i*dt) # interpolate between c_near and c_end
        q_end, valid = computeqgrasppose(robot, q_prev, cube, c) # compute the new configuration

        if not valid:
            # if a collision is detected in the new configuration, return the previous configuration and cube placement
            return q_prev, c_prev
        
        # update the previous configuration and cube placement
        q_prev = q_end
        c_prev = c

    return q_end, c_end


def valid_edge_to_goal(robot, cube, q_new, c_new, c_goal, discretisationsteps, delta_q):
    '''
    Returns True if the edge between q_new and c_new and the goal c_goal is collision free.
    An error smaller than EPSILON is considered as a valid edge.
    '''
    return norm(c_goal.translation - new_placement(robot, cube, q_new, c_new, c_goal, discretisationsteps, delta_q)[1].translation) < EPSILON


def RRT(robot, cube, q_init, q_goal, cubeplacementq0, cubeplacementqgoal, k=5000, delta_q=0.05):
    '''
    RRT algorithm to find a path from q_init to q_goal.
    The algorithm runs for k iterations and returns the graph G and a boolean indicating if a path was found.
    '''

    # set the number of discretisation steps for interpolation
    discretisationsteps_newconf = 200
    discretisationsteps_validedge = 200

    # initialise initial and goal configurations and cube placements
    q_init = q_init.copy()
    q_goal = q_goal.copy()
    c_init = cubeplacementq0
    c_goal = cubeplacementqgoal

    # initialise the connected graphs G_start and G_end
    G = [(None, q_init, c_init)]

    for i in range(k):
        print("Iteration", i)

        c_rand = generate_random_cube_placement() # generate a random cube placement

        # get the nearest vertex in the graph to the random configuration
        c_near_idx = nearest_vertex(G, c_rand)
        q_near, c_near = G[c_near_idx][1], G[c_near_idx][2]

        # compute a new configuration and cube placement by interpolating between c_near and c_rand
        q_new, c_new = new_placement(robot, cube, q_near, c_near, c_rand, discretisationsteps_newconf, delta_q)

        # add the new configuration and cube placement to the graph
        add_edge_and_vertex(G, c_near_idx, q_new, c_new)

        # check if the new configuration is close to the goal
        if valid_edge_to_goal(robot, cube, q_new, c_new, c_goal, discretisationsteps_validedge, delta_q):
            print("Path found")
            add_edge_and_vertex(G, len(G)-1, q_goal, c_goal)
            return G, True
        
    print("Path not found")

    return G, False

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


def plot_connected_graph(G):
    '''
    Plot the connected graph G in 3D.
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for _, _, c in G:
        ax.scatter(c.translation[0], c.translation[1], c.translation[2], c='blue', marker='o')

    path1 = get_path(G, with_cube=True)

    for i in range(len(path1)-1):
        _, c1 = path1[i]
        _, c2 = path1[i+1]
        ax.plot([c1.translation[0], c2.translation[0]], 
                [c1.translation[1], c2.translation[1]], 
                [c1.translation[2], c2.translation[2]], 'r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set the title of the plot to the index of the iteration
    ax.set_title(f'Graph coverage and path generated')

    # set the limits of the plot so that it is always in the same scale
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    # save image in img directory with name of index i
    fig.savefig(f'img/vanilla_path_plot.png')

    plt.close(fig)


def computepath(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, k=5000, delta_q=0.05):
    '''
    Compute a collision-free path from qinit to qgoal under grasping constraints.
    The path is expressed as a list of configurations.
    '''
    # run the RRT-Connect algorithm to find a path between qinit and qgoal
    G, pathfound = RRT(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, k, delta_q)
    
    # plot the connected graph
    plot_connected_graph(G)
    
    if not pathfound:
        return None, G

    path = get_path(G)

    return path


def displaypath(robot,path,dt,viz):
    if path is None:
        return

    setcubeplacement(robot, cube, CUBE_PLACEMENT)
    for q in path:
        viz.display(q)
        time.sleep(dt)

    # for q, c in path:
    #     setcubeplacement(robot, cube, c)
    #     viz.display(q)
    #     time.sleep(dt)

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