import path
import rrt
import path_rrt_connect

from tools import setupwithmeshcat

from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from inverse_geometry import computeqgrasppose

if __name__ == "__main__":

    # run the computepath from rrt, path, path_rrt_connect with different values of delta_q starting from 0.005, 0.01, 0.05, 0.1, 0.2
    
    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz=None)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz=None)

    if not(successinit and successend):
        print ("error: invalid initial or end configuration")

    # 0.005, 0.01, 
    # take the avearege of itereations for 20 runs

    for delta_q in [0.05, 0.1, 0.2]:
        avg = {
            "rrt_connect": 0,
            "rrt": 0,
            "rrt_star": 0
        }
        for i in range(20):
            print(f"delta_q = {delta_q}")
            # rrt connect
            pathFound, G, i = path_rrt_connect.computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, k=20000, delta_q=delta_q)
            print(i)
            avg["rrt_connect"] += i
            # rrt
            pathFound, G, i = path.computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, k=20000, delta_q=delta_q)
            avg["rrt"] += i
            print(i)
            # rrt *
            pathFound, G, i = rrt.computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, k=20000, delta_q=delta_q)
            avg["rrt_star"] += i
            print(i)
        avg["rrt_connect"] /= 20
        avg["rrt"] /= 20
        avg["rrt_star"] /= 20
        print(f"avg = {avg}")

    



# 0.005: 316, 5000, 5000
# 0.01: 1772, 5000, 1114
# 0.05: 312, 1910, 563
# 0.1: 312, 2065, 144
# 0.2: 62, 5000, 162