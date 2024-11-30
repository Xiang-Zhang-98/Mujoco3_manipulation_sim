# import sys
# import time
# from math import pi
# import numpy as np
# import matplotlib.pyplot as plt

# np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

# def trajectory_xyz(A, B, C, Hz=125):

#     time1 = time.time()

#     #start = np.array( [[ 14.0362, -7.5898, -16.3597, 0 ]], dtype=float )
#     #end = np.array( [[ -14.0362, -7.5898, -16.3597, 40 ]], dtype=float )
#     #A = np.array([[ 14.0362, -7.5898, -16.3597, 0 ]])
#     #B = np.array([[ -14.0362, -7.5898, -16.3597, 40 ]])
#     A1 = np.array([A[0, 0], A[0, 1], A[0, 2], A[0, 6]])
#     #print(A1)
#     A2 = np.array([A[0, 3], A[0, 4], A[0, 5], A[0, 6]])
#     B1 = np.array([B[0, 0], B[0, 1], B[0, 2], B[0, 6]])
#     B2 = np.array([B[0, 3], B[0, 4], B[0, 5], B[0, 6]])
#     C1 = np.array([C[:, 0], C[:, 1], C[:, 2], C[:, 6]])
#     C1 = np.transpose(C1)
#     #print(C1)
#     C2 = np.array([C[:, 3], C[:, 4], C[:, 5], C[:, 6]])
#     C2 = np.transpose(C2)

#     start = np.array([A1], dtype=float)
#     end = np.array([B1], dtype=float)
#     start1 = np.array([A2], dtype=float)
#     end1 = np.array([B2], dtype=float)

#     n = len(C)  # To know the number of rows
#     num_of_via = n
#     via = np.zeros((num_of_via, 4))

#     for i in range(num_of_via):
#         via[i, :] = start + (i + 1) / (num_of_via + 1) * (end - start)

#     #C = np.array([[ 14.0362, -5.1056, -27.5861,  via[0,3]],

# # [ -14.0362, -5.1056, -27.5861, via[1,3]]])
#     for i in range(n):
#         C2[i, 3] = C1[i, 3] = via[i, 3]

#     #C1[0,3] = via[0,3]
#     #C1[1,3] = via[1,3]
#     #C2[0,3] = via[0,3]
#     #C2[1,3] = via[1,3]

#     via = np.array(C1, dtype=float)
#     via1 = np.array(C2, dtype=float)

#     # print("via: ", via)
#     x = np.concatenate((start, via, end))
#     x1 = np.concatenate((start1, via1, end1))
#     #print(x)
#     command_list = np.zeros((via.shape[0], 7))

#     traj_pos, traj_vel, traj_acc, t = traj_planning_ALL(x, Hz)
#     traj_pos1, traj_vel1, traj_acc1, t = traj_planning_ALL(x1, Hz)
#     #print(traj_pos)
#     #for i in range(via.shape[0]):
#     #    for j in range(traj_pos.shape[0]):
#     #        if (via[i, :3]+0.001 > traj_pos[j, :3]).all() and (traj_pos[j, :3] > via[i, :3]-0.001).all():
#     # print(traj_pos[j, :3], traj_vel[j, :3], t[j])
#     #            command_list[i] = np.hstack( (traj_pos[j, :3], traj_vel[j, :3] ,t[j]) )

#     #return x, traj_pos

#     waypoints = np.hstack((np.delete(x, 3, 1), np.delete(x1, 3, 1)))
#     pos = np.hstack((traj_pos, traj_pos1))
#     vel = np.hstack((traj_vel, traj_vel1))
#     acc = np.hstack((traj_acc, traj_acc1))
#     return pos, vel, acc
#     sys.exit()

# def traj_planning_ALL(x, Hz):
#     """
#     input: every points(start + via + end)
#     """
#     n = x.shape[0]  # number of all x
#     v = np.zeros((n, 3))
#     for i in range(1, n - 1, 1):
#         v_last = (x[i, :3] - x[i - 1, :3]) / (x[i, 3] - x[i - 1, 3])
#         v_next = (x[i + 1, :3] - x[i, :3]) / (x[i + 1, 3] - x[i, 3])
#         for j in range(3):
#             if np.sign(v_last[j]) == np.sign(v_next[j]):
#                 v[i, j] = (v_last[j] + v_next[j]) / 2

#     traj_pos = traj_vel = traj_acc = t = np.array([[]])

#     for i in range(n - 1):
#         x1 = x[i, :3]
#         x2 = x[i + 1, :3]
#         v1 = v[i, :3]
#         v2 = v[i + 1, :3]
#         dt = x[i + 1, 3] - x[i, 3]

#         t_now = x[i + 1, 3] * Hz

#         t_num = int(round(t_now - traj_pos.shape[0] + 1))
#         if i == 0:
#             last_dt = 0
#         else:
#             last_dt = t[-1]

#         traj_pos_, traj_vel_, traj_acc_, t_ = traj_planning(
#             x1, v1, x2, v2, last_dt, dt, t_num)
#         if i == 0:
#             traj_pos = traj_pos_
#             traj_vel = traj_vel_
#             traj_acc = traj_acc_
#             t = t_
#         else:
#             # traj_pos = np.concatenate((traj_pos, traj_pos_))
#             # traj_vel = np.concatenate((traj_vel, traj_vel_))
#             # traj_acc = np.concatenate((traj_acc, traj_acc_))
#             traj_pos = np.concatenate((traj_pos, traj_pos_[1:]))
#             traj_vel = np.concatenate((traj_vel, traj_vel_[1:]))
#             traj_acc = np.concatenate((traj_acc, traj_acc_[1:]))
#             t = np.concatenate((t, t_[1:]))
#     # t = np.linspace(x[0, 3], x[-1, 3], traj_pos.shape[0])

#     return traj_pos, traj_vel, traj_acc, t

# def traj_planning(x1, v1, x2, v2, last_dt, dt, t_num):
#     # dt [sec]

#     # time_start = time.time()

#     # Theta = T_4x4 * A
#     '''
#     T = np.array( [[1, 0, 0, 0],
#                    [1, dt, dt**2, dt**3],
#                    [0, 1, 0, 0],
#                    [0, 1, 2*dt, 3*dt**2]] )

#     T_inv = inv(T)
#     '''
#     '''
#     T_inv = np.array( [[1, 0, 0, 0],
#                        [0, 0, 1, 0],
#                        [-3/dt**2, 3/dt**2, -2/dt, -1/dt],
#                        [2/dt**3, -2/dt**3, 1/dt**2, 1/dt**2]] )
#     '''
#     """
#     condition of start and end point [J1, J2, J3]
#     [theta_i, theta_f, d_theta_i, d_theta_f]
#     """
#     Theta = np.array([[x1[0], x2[0], v1[0],
#                        v2[0]], [x1[1], x2[1], v1[1], v2[1]],
#                       [x1[2], x2[2], v1[2], v2[2]]])
#     A = np.zeros((3, 4))  # coefficient of polyomials [J1, J2, J3]
#     # for i in range(3):
#     #     A[i,:] = np.dot(T_inv, Theta[i,:])
#     a00 = Theta[0, 0]
#     a01 = Theta[0, 2]
#     a02 = (-3 / dt**2) * Theta[0, 0] + (3 / dt**2) * Theta[0, 1] + (
#         -2 / dt) * Theta[0, 2] + (-1 / dt) * Theta[0, 3]
#     a03 = (2 / dt**3) * Theta[0, 0] + (-2 / dt**3) * Theta[0, 1] + (
#         1 / dt**2) * Theta[0, 2] + (1 / dt**2) * Theta[0, 3]

#     a10 = Theta[1, 0]
#     a11 = Theta[1, 2]
#     a12 = (-3 / dt**2) * Theta[1, 0] + (3 / dt**2) * Theta[1, 1] + (
#         -2 / dt) * Theta[1, 2] + (-1 / dt) * Theta[1, 3]
#     a13 = (2 / dt**3) * Theta[1, 0] + (-2 / dt**3) * Theta[1, 1] + (
#         1 / dt**2) * Theta[1, 2] + (1 / dt**2) * Theta[1, 3]

#     a20 = Theta[2, 0]
#     a21 = Theta[2, 2]
#     a22 = (-3 / dt**2) * Theta[2, 0] + (3 / dt**2) * Theta[2, 1] + (
#         -2 / dt) * Theta[2, 2] + (-1 / dt) * Theta[2, 3]
#     a23 = (2 / dt**3) * Theta[2, 0] + (-2 / dt**3) * Theta[2, 1] + (
#         1 / dt**2) * Theta[2, 2] + (1 / dt**2) * Theta[2, 3]

#     A = np.array([[a00, a01, a02, a03], [a10, a11, a12, a13],
#                   [a20, a21, a22, a23]])

#     t = np.linspace(0, dt, t_num)
#     t_return = np.linspace(last_dt, last_dt + dt, t_num)
#     num = t.size

#     x = np.zeros((num, 3))
#     v = np.zeros((num, 3))
#     a = np.zeros((num, 3))
#     """
#     theta = a0 + a1*t + a2*t**2 + a3*t**3
#     d_theta = a1 + 2*a2*t + 3*a3*t**2
#     dd_theta = 2*a2 + 6*a3*t
#     """
#     x[:, 0] = A[0][0] + A[0][1] * t + A[0][2] * t**2 + A[0][3] * t**3
#     x[:, 1] = A[1][0] + A[1][1] * t + A[1][2] * t**2 + A[1][3] * t**3
#     x[:, 2] = A[2][0] + A[2][1] * t + A[2][2] * t**2 + A[2][3] * t**3

#     v[:, 0] = A[0][1] + 2 * A[0][2] * t + 3 * A[0][3] * t**2
#     v[:, 1] = A[1][1] + 2 * A[1][2] * t + 3 * A[1][3] * t**2
#     v[:, 2] = A[2][1] + 2 * A[2][2] * t + 3 * A[2][3] * t**2

#     a[:, 0] = 2 * A[0][2] + 6 * A[0][3] * t
#     a[:, 1] = 2 * A[1][2] + 6 * A[1][3] * t
#     a[:, 2] = 2 * A[2][2] + 6 * A[2][3] * t

#     # time_end = time.time()
#     # print("time: %.4f msec" % ((time_end-time_start)*1000) )

#     return x, v, a, t_return

# def plot_traj(x, v, a, t):
#     """plot [theta, omega, alpha] of [J1, J2, J3]"""
#     x_axis_tick = 1
#     # plt.figure(figsize=(12.8, 9.6), dpi=100)
#     plt.figure(figsize=(12.8, 3.2), dpi=100)

#     plt.subplot(1, 3, 1)
#     plt.plot(t, x[:, 0], 'r', label="J1")
#     plt.plot(t, x[:, 1], 'g', label="J2")
#     plt.plot(t, x[:, 2], 'b', label="J3")
#     plt.legend(loc='upper right')
#     plt.axis([0, t[-1], -270, 270])
#     # plt.axis([0, t[-1], -10, 50])
#     plt.xticks(np.arange(0, t[-1], x_axis_tick))
#     plt.xlabel("time [sec]")
#     plt.ylabel(r"${\theta}$ [deg]")
#     plt.title("Joint Position")

#     plt.subplot(1, 3, 2)
#     plt.plot(t, v[:, 0], 'r', label="J1")
#     plt.plot(t, v[:, 1], 'g', label="J2")
#     plt.plot(t, v[:, 2], 'b', label="J3")
#     plt.legend(loc='upper right')
#     plt.axis([0, t[-1], -180, 180])
#     # plt.axis([0, t[-1], -20, 20])
#     plt.xticks(np.arange(0, t[-1], x_axis_tick))
#     plt.xlabel("time [sec]")
#     plt.ylabel(r"${\dot\theta}$ [deg/s]")
#     plt.title("Joint Velocity")

#     plt.subplot(1, 3, 3)
#     plt.plot(t, a[:, 0], 'r', label="J1")
#     plt.plot(t, a[:, 1], 'g', label="J2")
#     plt.plot(t, a[:, 2], 'b', label="J3")
#     plt.legend(loc='upper right')
#     plt.axis([0, t[-1], -360, 360])
#     # plt.axis([0, t[-1], -40, 40])
#     plt.xticks(np.arange(0, t[-1], x_axis_tick))
#     plt.xlabel("time [sec]")
#     plt.ylabel(r"${\ddot\theta}$ ${[deg/s^2]}$")
#     plt.title("Joint Acceleration")

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()
#     # test()
