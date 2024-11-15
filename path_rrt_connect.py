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

def generate_random_cube_placement(robot, cube, q_current):
    '''
    Generate a random cube placement within the constraints, and check for collisions
    '''
    # set the limits for the random cube placement
    x_min, x_max = 0.3, 0.42  # X-axis limits could be 0.33, 0.4
    y_min, y_max = -0.4, 0.14  # Y-axis limits could be -0.3, 0.11
    z_min, z_max = 0.93, 1.3  # Z-axis limits (keeping the cube above the obstacle)

    counter = 0

    while True:
        # sample a random cube placement uniformly within the constraints
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)

        placement = pin.SE3(rotate('z', 0), np.array([x, y, z]))

        return q_current, placement
        
    #     setcubeplacement(robot, cube, placement)

    #     q_current = q_current.copy()
    #     # Get robot configuration for the cube placement and check for collisions using computegrasppose
    #     q_rand, not_in_collision = computeqgrasppose(robot, q_current, cube, placement)

    #     if not_in_collision:
    #         return q_rand, placement
        
    #     counter += 1
    #     if counter > 100:
    #         break 

    # print("Error: Could not find a valid random configuration in 100 iterations")
    # return None

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
    c_end = c_rand.copy()
    dist = distance(c_near, c_rand)
    outcome = REACHED
    if delta_q is not None and dist > delta_q:
        c_end = lerp_cube(c_near, c_rand, delta_q/dist)
        outcome = ADVANCED
        # setcubeplacement(robot, cube, c_end)
        # q_end, valid = computeqgrasppose(robot, q_near, cube, c_end, viz=None)
        # viz.display(q_end)
        # input("Displaying c_end. Press Enter to continue")

    dt = 1 / discretisationsteps
    q_prev = q_near.copy()
    c_prev = c_near.copy()
    for i in range(1,discretisationsteps + 1):
        c = lerp_cube(c_near, c_end, i*dt)
        #print the type of the visualisation
        #print(type(visualisation))
        q_end, valid = computeqgrasppose(robot, q_prev, cube, c)
        # setcubeplacement(robot, cube, c)
        # viz.display(q_end)
        # print("valid:", valid)
        # print("collision:", collision(robot, q_end))
        if not valid:
            # print("Trapped at index", i)
            # answer = input("Press Enter to continue, q to quit")
            # if answer != "":
            outcome = TRAPPED
            return q_prev, c_prev, outcome
        q_prev = q_end
        c_prev = c
    return q_end, c_end, outcome

def extend(G, q_rand, c_rand, discretisationsteps, delta_q):
    c_near_idx = nearest_vertex(G, c_rand)
    q_near, c_near = G[c_near_idx][1], G[c_near_idx][2]
    # setcubeplacement(robot, cube, c_rand)
    # viz.display(q_rand)
    # input("Displaying q_rand. Press Enter to continue")
    q_new, c_new, outcome = new_placement(robot, cube, q_near, c_near, c_rand, discretisationsteps, delta_q)
    add_edge_and_vertex(G, c_near_idx, q_new, c_new)
    # print("OUTCOME:", outcome)
    # print("c_near:", c_near.translation)
    # print("c_rand:", c_rand.translation)
    # print("c_new:", c_new.translation)
    # print("c_new is being added to the graph")
    # count = 0
    # for _, _, cubesss in G:
    #     if norm(cubesss.translation - c_new.translation) == 0:
    #         count += 1
    # print("there were already ", count, "cubes in the graph\n")
        
    return q_new, c_new, outcome

def connect(G, q_rand, c_new, discretisationsteps, delta_q=0.01):
    '''
    Connects a new configuration c_new to the graph G by finding the nearest vertex and adding an edge.
    '''
    while True:
        # print("Running extend from connect")
        _, _, outcome = extend(G, q_rand, c_new, discretisationsteps, delta_q)
        if outcome != ADVANCED:
            break
    return outcome == REACHED

def swap(G1, G2):
    '''
    Swaps the two graphs G1 and G2
    '''
    return G2[:], G1[:]

def RRT_CONNECT(robot, cube, q_init, q_goal, cubeplacementq0, cubeplacementqgoal, k=5000, delta_q=0.05):

    discretisationsteps_newconf = 200

    q_init = q_init.copy()
    q_goal = q_goal.copy()
    c_init = cubeplacementq0
    c_goal = cubeplacementqgoal
    G_start = [(None, q_init, c_init)]
    G_end = [(None, q_goal, c_goal)]

    for i in range(0, k, 2):
        print("Iteration", i)

        q_rand, c_rand = generate_random_cube_placement(robot, cube, q_init)
        q_new, c_new, _ = extend(G_start, q_rand, c_rand, discretisationsteps_newconf, delta_q)

        if connect(G_end, q_new, c_new, discretisationsteps_newconf, delta_q):
            print("Path found")
            return G_start, G_end, True
        
        # plot_connected_graphs(G_start, G_end, i)
        
        print("Iteration", i+1)

        q_rand, c_rand = generate_random_cube_placement(robot, cube, q_goal)
        q_new, c_new, _ = extend(G_end, q_rand, c_rand, discretisationsteps_newconf, delta_q)

        if connect(G_start, q_new, c_new, discretisationsteps_newconf, delta_q):
            print("Path found")
            return G_start, G_end, True
        
        # plot_connected_graphs(G_start, G_end, i+1)
        

    print("Path not found")

    return G_start, G_end, False

def get_path(G):
    path = []
    node = G[-1]
    while node[0] is not None:
        path = [(node[1], node[2])] + path
        node = G[node[0]]
    path = [(G[0][1], G[0][2])] + path
    return path

def plot_connected_graphs(G1, G2, index=0):
    '''
    Plot the connected graphs G1 and G2 in 3D, each with a different color, and lines connecting the connected vertices.
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for _, _, c in G1:
        ax.scatter(c.translation[0], c.translation[1], c.translation[2], c='blue', marker='o')

    for _, _, c in G2:
        ax.scatter(c.translation[0], c.translation[1], c.translation[2], c='red', marker='o')

    path1 = get_path(G1)
    path2 = get_path(G2)

    for i in range(len(path1)-1):
        q1, c1 = path1[i]
        q2, c2 = path1[i+1]
        ax.plot([c1.translation[0], c2.translation[0]], [c1.translation[1], c2.translation[1]], [c1.translation[2], c2.translation[2]], 'b')

    for i in range(len(path2)-1):
        q1, c1 = path2[i]
        q2, c2 = path2[i+1]
        ax.plot([c1.translation[0], c2.translation[0]], [c1.translation[1], c2.translation[1]], [c1.translation[2], c2.translation[2]], 'r')




    print("len(path1):", len(path1))
    print("len(path2):", len(path2))
    # print("len(G1):", len(G1))
    # print("len(G2):", len(G2))
    # print("G_start:")
    # for i in range(len(G1)):
    #     print(G1[i][2].translation)
    # print("G_end:")
    # for i in range(len(G2)):
    #     print(G2[i][2].translation)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set the title of the plot to the index of the iteration
    ax.set_title(f'Iteration {index}')

    # set the limits of the plot so that it is always in the same scale
    x_min, x_max = 0.3, 0.42  # X-axis limits could be 0.33, 0.4
    y_min, y_max = -0.4, 0.14  # Y-axis limits could be -0.3, 0.11
    z_min, z_max = 0.93, 1.3  # Z-axis limits (keeping the cube above the obstacle)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    # plt.show()

    # save image in img directory with name of index i
    fig.savefig(f'img/{index}.png')

#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, k=5000, delta_q=0.05):
    # Your existing RRT and path planning logic
    # Make sure to use the passed `robot` and `cube` variables
    G1, G2, pathfound = RRT_CONNECT(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, k, delta_q)
    plot_connected_graphs(G1, G2, index=1001)
    
    if not pathfound:
        return None, G1, G2
    
    path1 = get_path(G1)
    print("path1:", len(path1))
    path2 = get_path(G2)
    print("path2:", len(path2))

    path = path1 + path2[::-1]

    return path#, G # TODO path should just be the list of configurations.


def displaypath(robot,path,dt,viz):
    if path is None:
        return
    for q, c in path:
        setcubeplacement(robot, cube, c)
        viz.display(q)
        time.sleep(dt)

    # setcubeplacement(robot, cube, CUBE_PLACEMENT)
    # for q in path:
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

    input("Press Enter to display the path")
    
    while True:
        displaypath(robot,path,dt=0.05,viz=viz) # you ll probably want to lower dt
        if input("Press Enter to display the path again, type 'q' to quit") == 'q':
            break

    input("Press Enter to plot the path in 3D")
    # plot_trajectory_in_3D(path, G, displayG=True)