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

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    
    # locate the cube at the desired position
    setcubeplacement(robot, cube, cubetarget)

    q = qcurrent.copy()

    # set DT for the integration
    DT = 1e-1

    
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
            return None, False

    return q, not collision(robot, q)
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()

    random_q = np.random.rand(robot.model.nq)

    viz.display(random_q)
    import time
    time.sleep(3)
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)

    updatevisuals(viz, robot, cube, qe)