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

def RAND_CONF(q_current):
    #return a random configuration
    x_min, x_max = 0.33, 0.4  # X-axis limits could be 0.33, 0.4
    y_min, y_max = -0.3, 0.11  # Y-axis limits could be -0.3, 0.11
    z_min, z_max = 0.94, 1.1  # Z-axis limits (keeping the cube above the obstacle)

    counter = 0

    while True:
        # Generate a random cube placement within the constraints
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)

        placement = pin.SE3(rotate('z', 0), np.array([x, y, z]))
        
        setcubeplacement(robot, cube, placement)

        # Get robot configuration for the cube placement and check for collisions using computegrasppose
        q_rand = q_current.copy()

        q_rand, not_in_collision = computeqgrasppose(robot, q_rand, cube, placement)

        if not_in_collision:
            return q_rand
        
        counter += 1
        if counter > 100:
            break 

    print("Error: Could not find a valid random configuration")
    return None

def distance(q1,q2):
    return norm(q1-q2)

def NEAREST_VERTEX(G, q_rand):
    #find the vertex in G that is closest to q_rand
    min_dist = np.inf
    nearest_vertex = None
    for (i,(parent, config)) in enumerate(G):
        dist = distance(config, q_rand)
        if dist < min_dist:
            min_dist = dist
            nearest_vertex = i
    return nearest_vertex

def ADD_EDGE_AND_VERTEX(G, parent, q):
    G += [(parent, q)]

def lerp(q0, q1, t):
    return q0*(1-t) + q1*t

def NEW_CONF(q_near, q_rand, discretisationsteps, delta_q=None):
    q_end = q_rand.copy()
    dist = distance(q_near, q_rand)
    if delta_q is not None and dist > delta_q:
        # if the distance between q_near and q_rand is greater than delta_q, 
        # we need get a new configuration q_end that is delta_q away from q_near
        # we use delta_q/dist to get the ratio of the distance between q_near and q_rand
        q_end = lerp(q_near, q_rand, delta_q/dist)
        dist = delta_q

    dt = dist / discretisationsteps
    for i in range(1,discretisationsteps):
        q = lerp(q_near, q_end, i*dt/dist)
        if collision(robot, q):
            return lerp(q_near, q_end, (i-1)*dt)
    return q_end

def VALID_EDGE(q_new, q_goal, discretisationsteps):
    return norm(q_goal - NEW_CONF(q_new, q_goal, discretisationsteps)) < EPSILON

def rrt(q_init, q_goal, k=1000, delta_q=0.1):

    discretisationsteps_newconf = 200
    discretisationsteps_validedge = 200

    q_init = q_init.copy()
    q_goal = q_goal.copy()
    G = [(None, q_init)]
    for _ in range(k):
        q_rand = RAND_CONF(q_init)
        q_near_idx = NEAREST_VERTEX(G, q_rand)
        q_near = G[q_near_idx][1]
        q_new = NEW_CONF(q_near, q_rand, discretisationsteps_newconf, delta_q)
        ADD_EDGE_AND_VERTEX(G, q_near_idx, q_new)
        if VALID_EDGE(q_new, q_goal, discretisationsteps_validedge):
            print("Path found")
            ADD_EDGE_AND_VERTEX(G, len(G)-1, q_goal)
            return G, True
    print("Path not found")
    return G, False

def get_path(G):
    path = []
    node = G[-1]
    while node[0] is not None:
        path = [node[1]] + path
        node = G[node[0]]
    path = [G[0][1]] + path
    return path

#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):

    G, pathfound = rrt(qinit, qgoal)
    if not pathfound:
        return None
    
    path = get_path(G)
    
    return path


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, TABLE_PLACEMENT, OBSTACLE_PLACEMENT
    from inverse_geometry import computeqgrasppose
    
    
    robot, cube, viz = setupwithmeshcat()
    

    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz=None)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz=None)
    print("WE HAVE A PATH")
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
    
