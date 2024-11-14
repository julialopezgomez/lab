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

def generate_random_cube_placement(robot, cube, q_current):
    '''
    Generate a random cube placement within the constraints, and check for collisions
    '''
    # set the limits for the random cube placement
    x_min, x_max = 0.33, 0.4  # X-axis limits could be 0.33, 0.4
    y_min, y_max = -0.3, 0.11  # Y-axis limits could be -0.3, 0.11
    z_min, z_max = 0.93, 1.1  # Z-axis limits (keeping the cube above the obstacle)

    counter = 0

    while True:
        # sample a random cube placement uniformly within the constraints
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)

        placement = pin.SE3(rotate('z', 0), np.array([x, y, z]))
        
        setcubeplacement(robot, cube, placement)

        q_current = q_current.copy()
        # Get robot configuration for the cube placement and check for collisions using computegrasppose
        q_rand, not_in_collision = computeqgrasppose(robot, q_current, cube, placement)

        if not_in_collision:
            return q_rand, placement
        
        counter += 1
        if counter > 100:
            break 

    print("Error: Could not find a valid random configuration in 100 iterations")
    return None

def distance(c1,c2):
    '''
    Returns the distance between two cube placements, 
    without considering the orientation
    '''
    return norm(c1.translation-c2.translation)

def nearest_vertex(G, c_rand):
    '''
    Finds the nearest vertex index in the graph G to the random configuration c_rand
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
    Add a new vertex to the grasph, including parent, configuration and cube placement.
    '''
    G += [(parent, q, c)]

def lerp(p0, p1, t):
    '''
    Linear interpolation between two points p0 and p1
    '''
    return p0*(1-t) + p1*t

def lerp_cube(cube_0, cube_1, t):
    '''
    Linear interpolation between two cube placements cube_0 and cube_1. Returns a new cube placement.
    '''
    new_placement = lerp(cube_0.translation, cube_1.translation, t)
    return pin.SE3(rotate('z', 0), new_placement)

def new_placement(robot, cube, q_near, c_near, c_rand, discretisationsteps, delta_q=None):
    '''
    Returns a new configuration q_end and a new cube placement c_end that is delta_q away from q_near and not in collision.
    If the new configuration is different from c_rand, the reached flag is set to False.
    '''
    c_end = c_rand.copy()
    dist = distance(c_near, c_rand)
    reached = True
    if delta_q is not None and dist > delta_q:
        # if the distance between q_near and q_rand is greater than delta_q, 
        # we need get a new configuration q_end that is delta_q away from q_near
        # we use delta_q/dist to get the ratio of the distance between q_near and q_rand
        c_end = lerp_cube(c_near, c_rand, delta_q/dist)
        reached = False

    dt = 1 / discretisationsteps
    q_prev = q_near.copy()
    c_prev = c_near.copy()
    for i in range(1,discretisationsteps + 1):
        c = lerp_cube(c_near, c_end, i*dt)
        q_end, valid = computeqgrasppose(robot, q_prev, cube, c)
        if not valid:
            return q_prev, c_prev, False # lerp_cube(c_near, c_end, (i-1)*dt)
        q_prev = q_end
        c_prev = c

    return q_end, c, reached

def valid_edge_to_goal(robot, cube, q_new, c_new, c_goal, discretisationsteps, delta_q=0.01):
    return norm(c_goal.translation - new_placement(robot, cube, q_new, c_new, c_goal, discretisationsteps, delta_q)[1].translation) < delta_q

def connect(G, c_new, discretisationsteps):
    '''
    Connects a new configuration c_new to the graph G by finding the nearest vertex and adding an edge.
    '''
    c_near_idx = nearest_vertex(G, c_new)
    q_near, c_near = G[c_near_idx][1], G[c_near_idx][2]
    q_new, c_new, reached = new_placement(robot, cube, q_near, c_near, c_new, discretisationsteps, delta_q = None)
    add_edge_and_vertex(G, c_near_idx, q_new, c_new)
    return reached

def RRT_CONNECT(robot, cube, q_init, q_goal, cubeplacementq0, cubeplacementqgoal, k=5000, delta_q=0.01):

    discretisationsteps_newconf = 200
    discretisationsteps_validedge = 200

    q_init = q_init.copy()
    q_goal = q_goal.copy()
    c_init = cubeplacementq0
    c_goal = cubeplacementqgoal
    G1 = [(None, q_init, c_init)]
    G2 = [(None, q_goal, c_goal)]

    for i in range(k):

        print("Iteration", i)
        _, c_rand = generate_random_cube_placement(robot, cube, q_init)
        c_near_idx = nearest_vertex(G1, c_rand)
        q_near, c_near = G1[c_near_idx][1], G1[c_near_idx][2]
        q_new, c_new, _ = new_placement(robot, cube, q_near, c_near, c_rand, discretisationsteps_newconf, delta_q)
        add_edge_and_vertex(G1, c_near_idx, q_new, c_new)

        if connect(G2, c_new, discretisationsteps_newconf):
            print("Path found")
            return G1, G2, True


        # if valid_edge_to_goal(robot, cube, q_new, c_new, c_goal, discretisationsteps_validedge, delta_q):
        #     print("Path found")
        #     add_edge_and_vertex(G, len(G)-1, q_goal, c_goal)
        #     return G, True
        
    print("Path not found")

    return G1, G2, False

def get_path(G1, G2):
    path = []
    node = G2[-1]
    while node[0] is not None:
        path += [node[1]]
        node = G2[node[0]]

    node = G1[G1[-1][0]]
    while node[0] is not None:
        path = [node[1]] + path
        node = G1[node[0]]
    
    return path

    

#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, k=5000, delta_q=0.01):
    # Your existing RRT and path planning logic
    # Make sure to use the passed `robot` and `cube` variables
    G1, G2, pathfound = RRT_CONNECT(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, k, delta_q)
    if not pathfound:
        return None, G1, G2

    path = get_path(G1, G2)

    return path#, G # TODO path should just be the list of configurations.


def displaypath(robot,path,dt,viz):
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

    input("Press Enter to display the path")
    
    while True:
        displaypath(robot,path,dt=0.05,viz=viz) # you ll probably want to lower dt
        if input("Press Enter to display the path again, type 'q' to quit") == 'q':
            break

    input("Press Enter to plot the path in 3D")
    # plot_trajectory_in_3D(path, G, displayG=True)