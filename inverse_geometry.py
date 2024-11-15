#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

def generate_random_cube_placement():
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

        return placement

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None, DT = 1e-1):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    
    # locate the cube at the desired position
    setcubeplacement(robot, cube, cubetarget)

    q = qcurrent.copy()

    # set DT for the integration

    
    counter = 0

    while True:
        counter += 1

        # compute the forward kinematics and the jacobians
        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.computeJointJacobians(robot.model, robot.data, q)

        # get the current position of the hands and the cube
        oMlhand = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
        oMrhand = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
        oMcubeL = getcubeplacement(cube, LEFT_HOOK)
        oMcubeR = getcubeplacement(cube, RIGHT_HOOK)

        # compute the error between the current position and the desired position for both hands
        lhandMcubeL = oMlhand.inverse() * oMcubeL
        rhandMcubeR = oMrhand.inverse() * oMcubeR

        # compute the spatial velocity form of the error for both hands
        lhand_nu = pin.log(lhandMcubeL).vector
        rhand_nu = pin.log(rhandMcubeR).vector
            
        # compute the jacobians of the hands
        lhand_J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(LEFT_HAND))
        rhand_J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(RIGHT_HAND))

        # compute the joint velocity that minimizes the error for each hand
        lhand_vq = pinv(lhand_J) @ lhand_nu # joint velocity for the left hand
        Plhand = np.eye(robot.model.nv) - pinv(lhand_J) @ lhand_J # null space projector for the left hand
        vq = lhand_vq + pinv(rhand_J @ Plhand) @ (rhand_nu - rhand_J @ lhand_vq) # joint velocity for the right hand

        # integrate the joint velocity to get the new configuration
        q = pin.integrate(robot.model, q, vq*DT)

        if viz is not None:
            viz.display(q)

        # converge if the error is smaller than EPSILON for both hands
        if norm(lhand_nu) < EPSILON and norm(rhand_nu) < EPSILON:
            break

        # throw an error if the algorithm did not converge
        if counter == 10000:
            print("Error: Could not find a valid grasping configuration in 10000 iterations")
            return None, False, counter

    return q, not collision(robot, q), counter
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()

    random_q = np.random.rand(robot.model.nq)

    viz.display(random_q)
    import time
    time.sleep(3)

    # create an array of 20 random configuration using the random_q and store them into an array
    rand_configs = []

    for i in range(1000):
        random_q = np.random.rand(robot.model.nq)
        rand_configs.append(random_q)
    
    q = robot.q0.copy()
    # run computeqgrasppose for successinit for 10 interval values between 0.001 and 1.0 50 times each and get the aferage number of iterations and standard deviation

    std = {
        "0.001": 0,
        "0.01": 0,
        "0.02": 0,
        "0.05": 0,
        "0.1": 0,
        "0.2": 0,
        "0.5": 0,
        "1.0": 0,
    }

    avg = {
        "0.001": 0,
        "0.01": 0,
        "0.02": 0,
        "0.05": 0,
        "0.1": 0,
        "0.2": 0,
        "0.5": 0,
        "1.0": 0,
    }

    for delta_q in [0.5, 1.0]:
        counters = []  # List to store counter values for each delta_q
        for i in range(len(rand_configs)):
            print(i)
            q0, successinit, counter = computeqgrasppose(robot, rand_configs[i], cube, CUBE_PLACEMENT, viz, DT=delta_q)
            counters.append(counter)  # Store each counter value

        # Calculate mean and standard deviation using numpy
        print("Delta_q: ", delta_q)
        print("Counters: ", counters)
        avg[str(delta_q)] = np.mean(counters)
        std[str(delta_q)] = np.std(counters)


    print(avg)
    print(std)
    
    q0,successinit, counter = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz, DT = 0.001)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz, DT = 0.001)

    updatevisuals(viz, robot, cube, qe)