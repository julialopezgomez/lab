#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np

from bezier import Bezier
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

def controllaw(sim, robot, trajs, tcurrent, cube):
    '''
    This function computes the torques to be applied to the robot in order to follow the trajectory trajs.
    The trajectory is a tuple of three functions q(t), vq(t), vvq(t) that return the desired configuration, velocity and acceleration at time t.
    '''
    q, vq = sim.getpybulletstate()
    
    #TODO compute the desired configuration, velocity and acceleration at time tcurrent 
    qd = trajs[0](tcurrent)
    vqd = trajs[1](tcurrent)
    vvqd = trajs[2](tcurrent)

    #TODO compute the error between the desired configuration and the current configuration
    e = qd - q
    #TODO compute the error between the desired velocity and the current velocity
    ed = vqd - vq
    
    #TODO compute the desired acceleration
    Kp*e + Kv*ed

    NLE coriollis force
    CRBA mass matrix






    torques = [0.0 for _ in sim.bulletCtrlJointsInPinOrder]
    sim.step(torques)

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    
    #setting initial configuration
    sim.setqsim(q0)
    
    
    #TODO this is just an example, you are free to do as you please.
    #In any case this trajectory does not follow the path 
    #0 init and end velocities
    def maketraj(path,T): #TODO compute a real trajectory !

        q_of_t = Bezier(path, t_max=T)
        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)
        return q_of_t, vq_of_t, vvq_of_t
    
    
    #TODO this is just a random trajectory, you need to do this yourself
    total_time=4.
    trajs = maketraj(path, total_time)   
    
    tcur = 0.
    
    
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
    
    
    