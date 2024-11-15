#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np

from bezier import Bezier

import pinocchio as pin

from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

def controllaw(sim, robot, trajs, tcurrent, cube):
    '''
    This function computes the torques to be applied to the robot in order to follow the trajectory trajs.
    The trajectory is a tuple of three functions q(t), vq(t), vvq(t) that return the desired configuration, velocity and acceleration at time t.
    '''
    q, vq = sim.getpybulletstate()
    
    # compute the desired configuration, velocity and acceleration at time tcurrent 
    qd = trajs[0](tcurrent)
    vqd = trajs[1](tcurrent)
    vvqd = trajs[2](tcurrent)

    # compute the forward kinematics and the jacobians
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.computeJointJacobians(robot.model, robot.data, q)

    # compute the jacobians for each of the end-effectors
    lhand_J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(LEFT_HAND))
    rhand_J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(RIGHT_HAND))

    # compute the desired forces for each of the end-effectors
    lforce = np.array([1., -35., 30., 0., 0., 0.])
    rforce = np.array([1., -35., 30., 0., 0., 0.])

    # compute the coriolis and centrifugal forces
    data = pin.Data(robot.model)
    coriolis = pin.nle(robot.model, data, q, vq)
    mass = pin.crba(robot.model, data, q)

    # compute the estimated joint acceleration
    vvq_star = vvqd + Kp * (qd - q) + Kv * (vqd - vq)

    # compute the torques to be applied to the robot
    torques = mass @ vvq_star + coriolis + lhand_J.T @ lforce + rhand_J.T @ rforce

    sim.step(torques)

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath, lerp
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    
    #setting initial configuration
    sim.setqsim(q0)

    
    def maketraj(path,T): 
        '''
        Takes a path and a total time T and returns a tuple of three functions 
        q(t), vq(t), vvq(t) that return the desired configuration, velocity and 
        acceleration at time t.
        '''
        interpolated_path = []
        number_of_interpolated_points = 1

        for i in range(len(path)-1):
            q0 = path[i]
            q1 = path[i+1]
            for i in range(number_of_interpolated_points):
                dt = 1/number_of_interpolated_points

                # linearly interpolate between q0 and q1
                interpolated_path.append(lerp(q0, q1, i*dt))
        
        interpolated_path.append(path[-1]) # add the last point

        segment_length = 4 # number of points in each segment

        # compute the number of Bezier segments
        num_bezier_segments = ((len(interpolated_path) - segment_length)//(segment_length-1)) + 1
        bezier_segments = [] 

        # number of extra points that are added to the last segment
        num_extra_points = len(interpolated_path) - ((segment_length-1)*num_bezier_segments + 1)

        # time fraction for each segment
        dt = (T/(len(interpolated_path)-1))*(segment_length-1)

        for i in range(num_bezier_segments):

            j = (segment_length-1)*i # start index of the segment
            control_points = []
            t_max = dt

            for k in range(segment_length):
                # add the control points of the Bezier curve
                control_points.append(interpolated_path[j+k])

            # if we are at the last segment, we need to add the extra points
            if i == num_bezier_segments-1:
                for k in range(num_extra_points):
                    control_points.append(interpolated_path[-num_extra_points+k])
                    # update the maximum time of the last segment
                    t_max = T - dt*i

            
            bezier_segments.append(Bezier(control_points, t_min=0, t_max=t_max))

        def q_of_t(t):
            segment = int(t//dt)
            # is segment is out of range, return the last segment
            if segment >= num_bezier_segments:
                segment = num_bezier_segments-1
            return bezier_segments[segment](t-dt*segment)
        
        def vq_of_t(t):
            segment = int(t//dt)
            # is segment is out of range, return the last segment
            if segment >= num_bezier_segments:
                segment = num_bezier_segments-1
            return bezier_segments[segment].derivative(1)(t-dt*segment)
        
        def vvq_of_t(t):
            segment = int(t//dt)
            # is segment is out of range, return the last segment
            if segment >= num_bezier_segments:
                segment = num_bezier_segments-1
            return bezier_segments[segment].derivative(2)(t-dt*segment)

        
        return q_of_t, vq_of_t, vvq_of_t
    
    
    total_time=4.
    trajs = maketraj(path, total_time)   
    
    tcur = 0.
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
    
    
    