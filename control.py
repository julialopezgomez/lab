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

    print(tcurrent)
    
    # compute the desired configuration, velocity and acceleration at time tcurrent 
    qd = trajs[0](tcurrent)
    vqd = trajs[1](tcurrent)
    vvqd = trajs[2](tcurrent)

    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.computeJointJacobians(robot.model, robot.data, q)

    lhand_J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(LEFT_HAND))
    rhand_J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(RIGHT_HAND))

    lforce = np.array([1., -30., 30., 0., 0., 0.])
    rforce = np.array([1., -30., 30., 0., 0., 0.])

    # magnitude tens, ignore angular

    data = pin.Data(robot.model)
    coriolis = pin.nle(robot.model, data, q, vq)
    mass = pin.crba(robot.model, data, q)

    vvq_star = vvqd + Kp * (qd - q) + Kv * (vqd - vq)

    torques = mass @ vvq_star + coriolis + lhand_J.T @ lforce + rhand_J.T @ rforce

    # NLE coriollis force
    # CRBA mass matrix

    # torques = [0.0 for _ in sim.bulletCtrlJointsInPinOrder]
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
    # path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, k=5000, delta_q=0.01)

    path =  [np.array([-4.93760518e-01, -7.90236986e-17, -1.80923671e-17, -7.81005451e-01,
        7.56016069e-02, -3.12380198e-01,  6.61367973e-19,  2.36778591e-01,
       -2.95236805e-01, -4.26308623e-02, -3.51693410e-01,  1.59275267e-01,
       -7.80012256e-18,  1.92418143e-01,  2.10797866e+00]), np.array([-4.91225979e-01, -7.91928080e-17, -1.78723199e-17, -7.73646690e-01,
        6.43583805e-02, -3.20934743e-01, -2.54826929e-18,  2.56576362e-01,
       -3.05927332e-01, -4.08060366e-02, -3.61940323e-01,  1.53168359e-01,
       -5.25060264e-18,  2.08771963e-01,  2.10441732e+00]), np.array([-4.90361838e-01, -7.86408454e-17, -1.99136711e-17, -7.51262265e-01,
        1.66154506e-02, -4.26621468e-01,  2.46287872e-20,  4.10006017e-01,
       -3.29175897e-01, -2.48520045e-02, -4.03511729e-01,  5.65992668e-02,
        3.53513238e-18,  3.46912462e-01,  2.08759915e+00]), np.array([-4.38787561e-01, -7.61552388e-17, -1.93140567e-17, -6.64915347e-01,
       -1.16316565e-01, -2.38930730e-01,  9.44334776e-19,  3.55247295e-01,
       -4.67097092e-01, -3.23810929e-02, -5.10903795e-01,  2.49144255e-01,
       -1.65564396e-18,  2.61759540e-01,  2.04355396e+00]), np.array([-4.43784537e-01, -7.49949922e-17, -1.94441537e-17, -6.51690602e-01,
       -1.97180951e-01, -3.03820201e-01,  1.12466881e-18,  5.01001151e-01,
       -4.75324861e-01, -2.63016972e-02, -5.80749464e-01,  1.96020939e-01,
       -5.86816311e-18,  3.84728524e-01,  2.04247154e+00]), np.array([-4.57821982e-01, -7.56763518e-17, -1.92398791e-17, -6.69878892e-01,
       -1.81121928e-01, -3.52137165e-01,  7.07438650e-18,  5.33259093e-01,
       -4.43099126e-01, -2.51184597e-02, -5.68759313e-01,  1.50410756e-01,
        1.15954582e-18,  4.18348557e-01,  2.05532575e+00]), np.array([-4.60397055e-01, -7.80794824e-17, -1.84912319e-17, -6.77008585e-01,
       -1.50382656e-01, -3.84542345e-01, -8.28193994e-18,  5.34925001e-01,
       -4.33394360e-01, -1.90210868e-02, -5.40109160e-01,  1.12177991e-01,
        1.62763034e-18,  4.27931168e-01,  2.05180345e+00]), np.array([-4.81323662e-01, -8.02678347e-17, -1.72832847e-17, -7.12883199e-01,
       -1.26150063e-01, -5.53547106e-01,  2.03260909e-18,  6.79697169e-01,
       -3.76593139e-01,  5.00522968e-03, -5.13347580e-01, -6.01673730e-02,
       -8.79433885e-18,  5.73514953e-01,  2.04870374e+00]), np.array([-4.31005666e-01, -8.11279875e-17, -1.31605504e-17, -6.84439758e-01,
        6.49734066e-04, -6.77158165e-01, -1.05105407e-18,  6.76508431e-01,
       -4.55354576e-01,  6.35632688e-02, -3.77062160e-01, -2.37087874e-01,
       -3.63580310e-19,  6.14150035e-01,  1.93982770e+00]), np.array([-3.70781424e-01, -7.21249207e-17, -1.47867158e-17, -6.47073187e-01,
        1.05328597e-01, -7.88936369e-01,  4.66149265e-18,  6.83607772e-01,
       -5.52945389e-01,  1.29318733e-01, -2.53957893e-01, -4.05702611e-01,
       -2.53489433e-18,  6.59660503e-01,  1.81384800e+00]), np.array([-3.13579053e-01, -7.65635269e-17, -1.32469620e-17, -6.12365637e-01,
        2.14246931e-01, -8.36715286e-01, -1.95742446e-18,  6.22468355e-01,
       -6.44855310e-01,  1.80411672e-01, -1.24763742e-01, -5.10966429e-01,
       -1.66309852e-19,  6.35730170e-01,  1.70555269e+00]), np.array([-2.43351321e-01, -7.46680019e-17, -1.14387875e-17, -4.71582744e-01,
        1.28266438e-01, -7.69955742e-01, -7.78397383e-19,  6.41689304e-01,
       -8.55865935e-01,  2.45218461e-01, -1.35857696e-01, -5.04946478e-01,
       -1.24842227e-19,  6.40804174e-01,  1.57051817e+00]), np.array([-1.69525587e-01, -8.27032273e-17, -1.03598402e-17, -3.96571982e-01,
        1.78596974e-01, -8.13021593e-01,  4.69543204e-18,  6.34424619e-01,
       -1.00470243e+00,  3.36476586e-01, -2.20774292e-02, -6.26214297e-01,
       -2.43515823e-18,  6.48291726e-01,  1.40543431e+00]), np.array([-9.15468566e-02, -8.49345113e-17, -1.05447737e-17, -2.83190719e-01,
        1.15014398e-01, -7.76439592e-01, -3.22691399e-18,  6.61425194e-01,
       -1.19606242e+00,  4.28074641e-01,  5.03146105e-03, -6.72380096e-01,
       -7.97449084e-19,  6.67348635e-01,  1.23585752e+00]), np.array([-3.46858700e-02, -8.36443011e-17, -1.18235934e-17, -2.08110936e-01,
        1.00915683e-01, -7.15122672e-01,  6.42043644e-18,  6.14206989e-01,
       -1.32800319e+00,  4.91839489e-01,  5.59168857e-02, -6.73197216e-01,
       -1.76842778e-18,  6.17280331e-01,  1.11523169e+00]), np.array([ 1.89384938e-02, -8.82320677e-17, -1.23226393e-17, -1.48932352e-01,
        1.33680149e-01, -7.00213080e-01, -1.89572000e-18,  5.66532931e-01,
       -1.44080614e+00,  5.78150645e-01,  1.49957527e-01, -7.14538889e-01,
       -2.07218532e-18,  5.64581362e-01,  9.75296168e-01]), np.array([ 7.52032463e-02, -8.72289702e-17, -1.51416553e-17, -9.01450039e-02,
        3.76659709e-02, -7.16520268e-01, -4.59828616e-19,  6.78854297e-01,
       -1.55585824e+00,  6.61307809e-01,  1.15087941e-01, -7.88595523e-01,
       -2.76457135e-18,  6.73507583e-01,  8.35874252e-01]), np.array([ 6.69232187e-02, -8.72580096e-17, -2.01465637e-17, -9.90730885e-02,
       -1.35495701e-02, -7.00473026e-01, -7.17365480e-20,  7.14022596e-01,
       -1.53865013e+00,  6.29797964e-01,  5.25201380e-02, -7.65347404e-01,
        2.04587753e-17,  7.12827266e-01,  8.75664124e-01]), np.array([ 8.27752161e-02, -8.19607535e-17, -2.32836265e-17, -6.63681693e-02,
       -2.13471895e-01, -5.26694487e-01, -4.64278308e-19,  7.40166381e-01,
       -1.58720705e+00,  5.71869177e-01, -1.40812311e-01, -6.11537949e-01,
        8.21579658e-18,  7.52350260e-01,  9.17740914e-01]), np.array([ 6.73864916e-02, -8.20785271e-17, -2.55831909e-17, -7.56977516e-02,
       -1.33530752e-01, -4.97373370e-01, -7.38107344e-20,  6.30904123e-01,
       -1.56248874e+00,  5.52853972e-01, -7.13612366e-02, -5.65144728e-01,
       -3.43234676e-18,  6.36505965e-01,  9.52144844e-01]), np.array([ 5.72206859e-02, -8.07872044e-17, -2.91470210e-17, -8.31078399e-02,
       -4.16981390e-02, -4.59781811e-01,  1.85150373e-19,  5.01479950e-01,
       -1.54491285e+00,  5.46402280e-01,  1.31478776e-02, -5.15272077e-01,
        7.69777291e-19,  5.02124199e-01,  9.68762342e-01]), np.array([ 1.81932049e-01, -7.21445858e-17, -1.51443813e-17, -1.61172616e-01,
       -3.75376084e-02, -2.00881947e-01,  2.02953870e-19,  2.38419555e-01,
       -1.59076221e+00,  4.81647660e-01,  1.55882125e-01, -3.85597606e-01,
       -5.12989091e-18,  2.29715481e-01,  9.08007567e-01])]

    
    #setting initial configuration
    sim.setqsim(q0)
    
    
    #TODO this is just an example, you are free to do as you please.
    #In any case this trajectory does not follow the path 
    #0 init and end velocities

    def lerp(q0, q1, t):
        return q0*(1-t) + q1*t
    
    def maketraj(path,T): 
        # interpolated_path = []


        # for i in range(len(path)-1):
        #     q0 = path[i]
        #     q1 = path[i+1]
        #     for i in range(6):
        #         dt = 1/6
        #         interpolated_path.append(lerp(q0, q1, i*dt))
        # interpolated_path.append(path[-1])

        # segment_length = 3
        # num_bezier_segments = len(interpolated_path)//segment_length
        # bezier_segments = []
        # dt = T/num_bezier_segments
        # for i in range(num_bezier_segments):
        #     j = segment_length*i
        #     control_points = []
        #     for k in range(segment_length):
        #         control_points.append(interpolated_path[j+k])

        #     if i == num_bezier_segments-1:
        #         num_extra_points = len(interpolated_path) - segment_length*num_bezier_segments
        #         for k in range(num_extra_points):
        #             control_points.append(interpolated_path[-num_extra_points+k])
            
        #     bezier_segments.append(Bezier(control_points, t_min=i*dt, t_max=(i+1)*dt))

        # def q_of_t(t):
        #     segment = int(t//dt)
        #     return bezier_segments[segment](t)
        
        # def vq_of_t(t):
        #     segment = int(t//dt)
        #     return bezier_segments[segment].derivative(1)(t)
        
        # def vvq_of_t(t):
        #     segment = int(t//dt)
        #     return bezier_segments[segment].derivative(2)(t)

        q_of_t = Bezier(path, t_max=T)
        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = q_of_t.derivative(2)

        
        return q_of_t, vq_of_t, vvq_of_t
    
    
    #TODO this is just a random trajectory, you need to do this yourself
    total_time=4.
    trajs = maketraj(path, total_time)   
    
    tcur = 0.
    
    
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
    
    
    