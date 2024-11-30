import os
import time
import math
import numpy as np
from math import sin, cos, pi, atan2, asin, acos, sqrt
from numpy import cross
# import trajectory_cubic as traj
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection, Path3DCollection
import matplotlib.animation as animation

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


def IK(vec):
    """
    6-DOF,
    original mechanism,
    Geometric Solution
    Inverse Kinematics function
    """
    no_sol = False
    # (px, py, pz, rx, ry, rz) = (vec[0], vec[1], vec[2], vec[3]/180*math.pi, vec[4]/180*math.pi, vec[5]/180*math.pi)
    (px, py, pz, rx, ry, rz) = (vec[0], vec[1], vec[2], vec[3], vec[4], vec[5])
    
    Rx = np.array([ [1, 0, 0],
                    [0, cos(rx), -sin(rx)],
                    [0, sin(rx), cos(rx)] ])
    Ry = np.array([ [cos(ry), 0, sin(ry)],
                    [0, 1, 0],
                    [-sin(ry), 0, cos(ry)] ])
    Rz = np.array([ [cos(rz), -sin(rz), 0],
                    [sin(rz), cos(rz), 0],
                    [0, 0, 1] ])
    rot = np.dot(Rz, np.dot(Ry,Rx))
    #print('rot:', rot)

    # DH-parameters
    d = np.array([0.3300, 0.0000, 0.0000, -0.4200, 0.0000, -0.0800]) 
    al = np.array([-np.pi/2, np.pi, np.pi/2, -np.pi/2, np.pi/2, np.pi])
    a = np.array([0.0500, 0.4400, -0.035, 0, 0, 0])
    q = np.zeros(6)
    """
    Calculate theta1
    """
    t1 = np.zeros(2)
    T04_p = np.array([ px - np.abs(d[5])*rot[0,2], py - np.abs(d[5])*rot[1,2], pz - np.abs(d[5])*rot[2,2] ])
    #print('T04_p:', T04_p)
    t1 = np.array([ atan2(T04_p[1], T04_p[0]), atan2(T04_p[1], T04_p[0]) + math.pi ])
    #print('theta1',t1*180/math.pi)
    """
    Calculate theta3
    """
    q[0] = t1[0]
    T01 = np.array([[ math.cos(q[0]),  -math.sin(q[0])*math.cos(al[0]),  math.sin(q[0])*math.sin(al[0]),  a[0]*math.cos(q[0])],
                    [ math.sin(q[0]),   math.cos(q[0])*math.cos(al[0]), -math.cos(q[0])*math.sin(al[0]),  a[0]*math.sin(q[0])],
                    [              0,                  math.sin(al[0]),                 math.cos(al[0]),                 d[0]],
                    [              0,                                0,                               0,                   1]] )
    R01 = np.array([[ math.cos(q[0]),  -math.sin(q[0])*math.cos(al[0]),  math.sin(q[0])*math.sin(al[0])],
                    [ math.sin(q[0]),   math.cos(q[0])*math.cos(al[0]), -math.cos(q[0])*math.sin(al[0])],
                    [              0,                  math.sin(al[0]),                 math.cos(al[0])]] )
    P01 = np.array([T01[0,3],T01[1,3],T01[2,3]])
    #print('T01_p:',T01[0,3], T01[1,3], T01[2,3])
    T14_p = np.array([T04_p[0]-T01[0,3],T04_p[1]-T01[1,3],T04_p[2]-T01[2,3]])
    l2 = sqrt(T14_p[0]*T14_p[0]+T14_p[1]*T14_p[1]+T14_p[2]*T14_p[2])
    l1 = sqrt(a[2]*a[2]+d[3]*d[3])
    phi = acos((l1*l1+a[1]*a[1]-l2*l2)/(2*l1*a[1]))
    alpha = acos((-a[2])/l1)
    t3 = np.array([-np.pi+phi+alpha, -np.pi-phi+alpha])
    #t3 = np.array([np.pi-phi-alpha, -np.pi-phi+alpha])
    #print('theta3:',t3*180/np.pi)
    """
    Calculate theta2
    """
    beta1 = atan2(T14_p[2],sqrt(T14_p[0]*T14_p[0]+T14_p[1]*T14_p[1]))
    beta2 = asin((l1*l1+a[1]*a[1]-l2*l2)/(2*a[1]*l1)) + asin((l2*l2+l1*l1-a[1]*a[1])/(2*l1*l2))
    if T04_p[2] >= T01[2,3]:
        t2 = np.array([np.pi/2-(np.abs(beta1)+beta2), np.pi/2+(np.abs(beta1)-beta2)])
    else:
        t2 = np.array([np.pi/2+(np.abs(beta1)-beta2), np.pi/2-(np.abs(beta1)+beta2)])
    #print('theta2',t2*180/np.pi)
    """
    Calculate theta5
    """
    q[0] = t1[0]
    q[1] = t2[0]-math.pi/2
    q[2] = t3[0]+math.pi
    q[3] = 0
    T12 = np.array( [[ math.cos(q[1]),  -math.sin(q[1])*math.cos(al[1]),  math.sin(q[1])*math.sin(al[1]),  a[1]*math.cos(q[1])],
                     [ math.sin(q[1]),   math.cos(q[1])*math.cos(al[1]), -math.cos(q[1])*math.sin(al[1]),  a[1]*math.sin(q[1])],
                     [              0,                  math.sin(al[1]),                 math.cos(al[1]),                 d[1]],
                     [              0,                                0,                               0,                   1]] )
    T23 = np.array( [[ math.cos(q[2]),  -math.sin(q[2])*math.cos(al[2]),  math.sin(q[2])*math.sin(al[2]),  a[2]*math.cos(q[2])],
                     [ math.sin(q[2]),   math.cos(q[2])*math.cos(al[2]), -math.cos(q[2])*math.sin(al[2]),  a[2]*math.sin(q[2])],
                     [              0,                  math.sin(al[2]),                 math.cos(al[2]),                 d[2]],
                     [              0,                                0,                               0,                   1]] )
    T34 = np.array( [[ math.cos(q[3]),  -math.sin(q[3])*math.cos(al[3]),  math.sin(q[3])*math.sin(al[3]),  a[3]*math.cos(q[3])],
                     [ math.sin(q[3]),   math.cos(q[3])*math.cos(al[3]), -math.cos(q[3])*math.sin(al[3]),  a[3]*math.sin(q[3])],
                     [              0,                  math.sin(al[3]),                 math.cos(al[3]),                 d[3]],
                     [              0,                                0,                               0,                   1]] )

    T03 = np.array(np.dot(T01, np.dot(T12, T23)))
    t5 = acos(rot[0,2]*(-T03[0,2])+rot[1,2]*(-T03[1,2])+rot[2,2]*(-T03[2,2]))
    if rot[2,2] <= 0-0.000001 and rot[2,2] >= 0+0.000001:
        t5 = 0
    elif rot[2,2] < 0-0.000001:
        t5 = -t5 # end-effector point down
    #t5 = math.pi/2-acos(rot[0,2]*T04[0,1]+rot[1,2]*T04[1,1]+rot[2,2]*T04[2,1])
    q[4] = t5
    #print('theta5:',t5*180/math.pi)
    """
    Calculate theta 4 6
    """
    R03 = np.array(T03[0:3, 0:3])
    R30 = np.transpose(R03)
    rott = np.array(rot)
    
    for i in range(3):
        for j in range(3):
            if rot[i,j] <= 0+0.00016 and rot[i,j] >= 0-0.00016:
                rot[i,j] = 0.0
            if R30[i,j] <= 0+0.00016 and R30[i,j] >= 0-0.00016:
                R30[i,j] = 0.0
    rott[:,0] = -rot[:,0]
    rott[:,2] = -rot[:,2]
    for i in range(3):
        for j in range(3):
            if rott[i,j] <= 0+0.0001 and rott[i,j] >= 0-0.0001:
                rott[i,j] = 0.0
    R36 = np.dot(R30, rott)

    for i in range(3):
        for j in range(3):
            if R36[i,j] <= 0+0.0001 and R36[i,j] >= 0-0.0001:
                R36[i,j] = 0.0
    t4 = atan2(R36[1,2]/math.sin(t5), R36[0,2]/math.sin(t5))
    if R36[2,0] == 0:
        t6 = atan2(R36[2,1]/math.sin(t5), R36[2,0]/math.sin(t5))
    else:    
        t6 = atan2(R36[2,1]/math.sin(t5), -R36[2,0]/math.sin(t5))
    """
    Reorganize q
    """
    q[1] = t2[0]
    q[2] = t3[0]
    q[3] = t4
    q[5] = t6
    for i in range(6):      ## There is singular point
        if math.isnan(q[i]):
            no_sol = True
            break
    
    if no_sol:
        return False
    else:
        q[5] = -q[5]
        return q

def FK(q_vec):
    """
    6-DOF,
    original mechanism,
    Forward Kinematics function,
    return [x,y,z] of end-effector
    """
    dof = 6
    # DH-parameters
    d = np.array([0.3300, 0.0000, 0.0000, -0.4200, 0.0000, -0.0800])        
    al = np.array([-np.pi/2, np.pi, np.pi/2, -np.pi/2, np.pi/2, np.pi])
    a = np.array([0.0500, 0.4400, -0.035, 0, 0, 0])
    theta = np.array([0, -90, 180, 0, 0, 180]) # From DH table
    q_vec = (q_vec + theta)
    # each joint angle
    q = np.zeros(dof)
    for i in range(dof):
        q[i] = q_vec[i]*math.pi/180

    T01 = np.array( [[ math.cos(q[0]),  -math.sin(q[0])*math.cos(al[0]),  math.sin(q[0])*math.sin(al[0]),  a[0]*math.cos(q[0])],
                     [ math.sin(q[0]),   math.cos(q[0])*math.cos(al[0]), -math.cos(q[0])*math.sin(al[0]),  a[0]*math.sin(q[0])],
                     [              0,                  math.sin(al[0]),                 math.cos(al[0]),                 d[0]],
                     [              0,                                0,                               0,                   1]] )
    T12 = np.array( [[ math.cos(q[1]),  -math.sin(q[1])*math.cos(al[1]),  math.sin(q[1])*math.sin(al[1]),  a[1]*math.cos(q[1])],
                     [ math.sin(q[1]),   math.cos(q[1])*math.cos(al[1]), -math.cos(q[1])*math.sin(al[1]),  a[1]*math.sin(q[1])],
                     [              0,                  math.sin(al[1]),                 math.cos(al[1]),                 d[1]],
                     [              0,                                0,                               0,                   1]] )
    T23 = np.array( [[ math.cos(q[2]),  -math.sin(q[2])*math.cos(al[2]),  math.sin(q[2])*math.sin(al[2]),  a[2]*math.cos(q[2])],
                     [ math.sin(q[2]),   math.cos(q[2])*math.cos(al[2]), -math.cos(q[2])*math.sin(al[2]),  a[2]*math.sin(q[2])],
                     [              0,                  math.sin(al[2]),                 math.cos(al[2]),                 d[2]],
                     [              0,                                0,                               0,                   1]] )
    T34 = np.array( [[ math.cos(q[3]),  -math.sin(q[3])*math.cos(al[3]),  math.sin(q[3])*math.sin(al[3]),  a[3]*math.cos(q[3])],
                     [ math.sin(q[3]),   math.cos(q[3])*math.cos(al[3]), -math.cos(q[3])*math.sin(al[3]),  a[3]*math.sin(q[3])],
                     [              0,                  math.sin(al[3]),                 math.cos(al[3]),                 d[3]],
                     [              0,                                0,                               0,                   1]] )
    T45 = np.array( [[ math.cos(q[4]),  -math.sin(q[4])*math.cos(al[4]),  math.sin(q[4])*math.sin(al[4]),  a[4]*math.cos(q[4])],
                     [ math.sin(q[4]),   math.cos(q[4])*math.cos(al[4]), -math.cos(q[4])*math.sin(al[4]),  a[4]*math.sin(q[4])],
                     [              0,                  math.sin(al[4]),                 math.cos(al[4]),                 d[4]],
                     [              0,                                0,                               0,                   1]] )
    T56 = np.array( [[ math.cos(q[5]),  -math.sin(q[5])*math.cos(al[5]),  math.sin(q[5])*math.sin(al[5]),  a[5]*math.cos(q[5])],
                     [ math.sin(q[5]),   math.cos(q[5])*math.cos(al[5]), -math.cos(q[5])*math.sin(al[5]),  a[5]*math.sin(q[5])],
                     [              0,                  math.sin(al[5]),                 math.cos(al[5]),                 d[5]],
                     [              0,                                0,                               0,                   1]] )
    T_each = np.array([T01, T12, T23, T34, T45, T56])
    # after for loop, T_each = [T01, T02, T03, T04, T05, T06]
    for i in range(6):
        if i > 0:
            T_each[i] = np.dot(T_each[i-1],T_each[i])

    ## The position of end-effector
    points = np.zeros(3)
    points = np.array([ T_each[5][0, 3], T_each[5][1, 3], T_each[5][2, 3] ])

    ## Orientation
    ry = math.atan2(-T_each[5][2,0], math.sqrt(T_each[5][0,0]*T_each[5][0,0]+T_each[5][1,0]*T_each[5][1,0]))
    if ry <= -np.pi/2+0.00000001 and ry >= -np.pi/2-0.00000001:
        rz = 0
        rx = -math.atan2(T_each[5][0,1], T_each[5][1,1])
    elif ry <= np.pi/2+0.00000001 and ry >= np.pi/2-0.00000001:
        rz = 0
        rx = math.atan2(T_each[5][0,1], T_each[5][1,1])
    else:
        rz = math.atan2(T_each[5][1,0]/math.cos(ry), T_each[5][0,0]/math.cos(ry))
        rx = math.atan2(T_each[5][2,1]/math.cos(ry), T_each[5][2,2]/math.cos(ry))
    r_end_effector = np.array([rx, ry, rz])
    sol = np.array([points[0], points[1], points[2], r_end_effector[0], r_end_effector[1], r_end_effector[2]])
   
    return sol

