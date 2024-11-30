import mujoco
import mujoco.viewer
import numpy as np
import time
import os
# import gym
# from gym import spaces,register
import gymnasium as gym
from gymnasium import spaces,register
import transforms3d.quaternions as trans_quat
import transforms3d.euler as trans_eul
import copy
from .source.lrmate_kine_base import IK
from ray.tune.registry import register_env

class Fanuc_dual_arm_base(gym.Env):
    def __init__(self, render=False, verbose=False):
        super(Fanuc_dual_arm_base, self).__init__()
        self.m = mujoco.MjModel.from_xml_path('envs/source/LRMate_200iD_dual_arm.xml')
        self.d = mujoco.MjData(self.m)
        self.num_robot = 2
        self.robot_y_offset = 0.8
        # joint control gains & variables
        self.joint_acc = np.zeros([self.num_robot,8])
        
        self.joint_kp = np.zeros([self.num_robot,8])
        self.joint_kp[0] = np.array([100, 100, 100, 100, 100, 100, 100, 100])
        self.joint_kp[1] = np.array([100, 100, 100, 100, 100, 100, 100, 100])
        self.joint_kd = 2 * np.sqrt(self.joint_kp)

        self.joint_pos = np.zeros([self.num_robot,6])
        self.joint_vel = np.zeros([self.num_robot,6])
        self.joint_acc = np.zeros([self.num_robot,6])
        # for two robots
        self.ref_joint = np.zeros([self.num_robot,8])
        self.ref_vel = np.zeros([self.num_robot,8])
        self.ref_acc = np.zeros([self.num_robot,8])

        # gripper initial pose
        self.gripper_pose = np.array([0.031]*self.num_robot)
        self.set_gripper(self.gripper_pose)

        # admittance control gains
        self.adm_kp = np.zeros([self.num_robot,6])
        self.adm_m = np.zeros([self.num_robot,6])
        self.adm_kd = np.zeros([self.num_robot,6])
        self.adm_pose_ref = np.zeros([self.num_robot,7])
        self.adm_vel_ref = np.zeros([self.num_robot,6])

        # assign init values
        self.adm_kp[0] = 10 * np.array([1, 1, 1, 1, 1, 1])
        self.adm_m[0] = 1 * np.array([1, 1, 1, 1, 1, 1])
        self.adm_kd[0] = 4 * np.sqrt(np.multiply(self.adm_kp[0], self.adm_m[0]))
        self.adm_kp[1] = 10 * np.array([1, 1, 1, 1, 1, 1])
        self.adm_m[1] = 1 * np.array([1, 1, 1, 1, 1, 1])
        self.adm_kd[1] = 4 * np.sqrt(np.multiply(self.adm_kp[1], self.adm_m[1]))

        self.HZ = 125

        # place holder for robot state
        self.pose_vel = np.zeros([self.num_robot,13])
        self.force_sensor_data = np.zeros([self.num_robot,6])
        self.force_offset = np.zeros([self.num_robot,6])
        self.jacobian = np.zeros([self.num_robot,6,6])

        # peg-in-hole task setting
        self.eef_offset = np.array([0.3-0.0341, 0.0, 0.0])
        self.force_frame_offset = np.array([[-1, 0, 0],
                                            [0, -1, 0],
                                            [0, 0, -1]])

        # set init robot pose:
        init_c_pose = np.array([0.5, 0.0, 0.27, 0.0, np.pi, np.pi/2])
        init_j_pose = IK(init_c_pose)
        self.d.qpos[:6] = init_j_pose
        self.d.qpos[8:8+6] = init_j_pose

        self.render = render
        self.verbose = verbose
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d)

        mujoco.mj_step(self.m, self.d)
        self.force_calibration()

    def force_calibration(self, H=100):
        """
        Calibrate force sensor reading
        H: force history horizon
        """
        force_history = np.zeros([H,self.num_robot, 6])
        self.sim_step()
        self.set_reference_traj(
            self.joint_pos, 0 * self.joint_vel, 0 * self.joint_acc
        )
        for _ in range(H):
            self.sim_step()
            force_history[_, 0, :] = self.force_sensor_data[0]
            force_history[_, 1, :] = self.force_sensor_data[1]
        self.force_offset[0] = np.mean(force_history[int(H / 2):,0,:], axis=0)
        self.force_offset[1] = np.mean(force_history[int(H / 2):,1,:], axis=0)

    def get_gemo_pose_vel(self, geom_name):
        geom_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        geom_pos = self.d.geom_xpos[geom_id]
        geom_rot_mat = self.d.geom_xmat[geom_id].reshape([3,3])
        geom_quat = trans_quat.mat2quat(geom_rot_mat)
        geom_vel = np.zeros(6)
        # compute jacobian
        jacp = np.zeros([3, self.m.nv])
        jacr = np.zeros([3, self.m.nv])
        mujoco.mj_jacGeom(self.m, self.d, jacp, jacr, geom_id)
        Full_jacobian = np.concatenate([jacp, jacr])
        # compute velocity
        geom_vel = np.dot(Full_jacobian, self.d.qvel)
        return geom_pos, geom_rot_mat, geom_vel, Full_jacobian
    
    def get_pose_vel(self):
        # get robot end-effector pose and velocity for two robots
        # robot 1 link6 pose and vel
        r1_link6_pos, r1_link6_rot_mat, r1_link6_vel, r1_jacobian = self.get_gemo_pose_vel("link6_r1")
        # offset the mat
        r1_link6_rot_mat = r1_link6_rot_mat @ np.array([[ 1.0000,  0.0008,  0.0003],
                                                        [-0.0008,  0.8373,  0.5467],
                                                        [ 0.0002, -0.5467,  0.8373]]).T
        # convert to quat
        r1_link6_quat = trans_quat.mat2quat(r1_link6_rot_mat)

        # robot 2 link6 pose and vel
        r2_link6_pos, r2_link6_rot_mat, r2_link6_vel, r2_jacobian = self.get_gemo_pose_vel("link6_r2")
        # offset the mat
        r2_link6_rot_mat = r2_link6_rot_mat @ np.array([[ 1.0000,  0.0008,  0.0003],
                                                        [-0.0008,  0.8373,  0.5467],
                                                        [ 0.0002, -0.5467,  0.8373]]).T
        # convert to quat
        r2_link6_quat = trans_quat.mat2quat(r2_link6_rot_mat)

        # combine pose and vel
        r1_pose_vel = np.concatenate([r1_link6_pos, r1_link6_quat, r1_link6_vel[:6]])
        r2_pose_vel = np.concatenate([r2_link6_pos, r2_link6_quat, r2_link6_vel[:6]])
        # update robot state
        self.pose_vel[0] = r1_pose_vel
        self.pose_vel[1] = r2_pose_vel
        self.jacobian[0,:,:] = r1_jacobian[:6, :6]
        self.jacobian[1,:,:] = r2_jacobian[:, 8:8+6]
        self.joint_pos[0] = self.d.qpos[:6].copy()
        self.joint_vel[0] = self.d.qvel[:6].copy()
        self.joint_pos[1] = self.d.qpos[8:8+6].copy()
        self.joint_vel[1] = self.d.qvel[8:8+6].copy()
    
    def get_force_sensor_data(self, robot_id=1):
        sensor_name = "force_ee" + "_r" + str(robot_id)
        # get force sensor data
        sensor_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)

        # Get address and dimension of the sensor
        adr = self.m.sensor_adr[sensor_id]
        dim = self.m.sensor_dim[sensor_id]
        force = np.copy(self.d.sensordata[adr:adr + dim])
        # get torque sensor data
        sensor_name = "torque_ee" + "_r" + str(robot_id)
        sensor_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        adr = self.m.sensor_adr[sensor_id]
        dim = self.m.sensor_dim[sensor_id]
        torque = np.copy(self.d.sensordata[adr:adr + dim])
        force_torque = np.concatenate([force, torque])

        return force_torque
    
    def update_force_sensor(self):
        self.force_sensor_data[0] = self.get_force_sensor_data(robot_id=1)
        self.force_sensor_data[1] = self.get_force_sensor_data(robot_id=2)
    
    def computed_torque_control(self):
        # Assuming self.m and self.d are equivalent to m and d in the C++ code
        # Dual arm robot version, ref_joint contains 2 robots' joint angles

        # Compute inverse dynamics forces
        mujoco.mj_rne(self.m, self.d, 0, self.d.qfrc_inverse)
        for i in range(self.m.nv):
            self.d.qfrc_inverse[i] += (self.m.dof_armature[i] * self.d.qacc[i] -
                                    self.d.qfrc_passive[i] -
                                    self.d.qfrc_constraint[i])

        # Error and error derivative for robot 1 and 2
        e = self.d.qpos[:16] - self.ref_joint.flatten()[:16]  # ref_joint should be defined or passed as an argument
        e_dot = self.d.qvel[:16] - self.ref_vel.flatten()[:16]  # ref_vel should be defined or passed as an argument

        # Control law components
        kve_dot = np.multiply(self.joint_kd.flatten(), e_dot)  # kv should be defined or passed as an argument
        kpe = np.multiply(self.joint_kp.flatten(), e)  # kp should be defined or passed as an argument
        inertial_pd = self.ref_acc.flatten() - kve_dot - kpe  # ref_acc should be defined or passed as an argument

        # Compute full inertia matrix
        M = np.zeros((self.m.nv, self.m.nv))
        mujoco.mj_fullM(self.m, M, self.d.qM)
        M_robot = M[:self.m.nu, :self.m.nu]  # Assuming self.m.nu is the number of actuators

        # Compute the control torques
        inertial_torque = np.dot(M_robot, inertial_pd)

        # Apply control and inverse dynamics torques
        self.d.ctrl[:16] = np.clip(inertial_torque + self.d.qfrc_inverse[:16],-5,5)

        # Set gripper control
        self.d.ctrl[6] = self.gripper_pose[0]  # gripper_pose should be defined or passed as an argument
        self.d.ctrl[7] = self.gripper_pose[0]
        self.d.ctrl[6+8] = self.gripper_pose[1]
        self.d.ctrl[7+8] = self.gripper_pose[1]
    
    def get_eef_pose_vel(self, link6_pose_vel):
        link6_pos = link6_pose_vel[:3]
        link6_rotm = trans_quat.quat2mat(link6_pose_vel[3:7])
        link6_vel = link6_pose_vel[7:]

        eef_pos = link6_pos + link6_rotm @ self.eef_offset
        eef_rotm = copy.copy(link6_rotm)

        # Let's forget kinematics and assume eef and link 6 have the same vel
        # OH we cannot do that
        eef_vel = copy.copy(link6_vel)
        eef_vel[:3] = link6_vel[:3] + self.skew_symmetric(link6_vel[3:6]) @ link6_rotm @ self.eef_offset
        return eef_pos, eef_rotm, eef_vel
    
    def Cartersian_vel_control(self, vel):
        target_joint_vel = np.linalg.pinv(self.jacobian) @ vel
        return target_joint_vel

    def admittance_control(self):
        # admittance control for two robots
        target_joint_vel = np.zeros([self.num_robot,6])
        for i in range(self.num_robot):
            ## Get robot motion from desired dynamics
            eef_pos, eef_rotm, eef_vel = self.get_eef_pose_vel(self.pose_vel[i])
            eef_pos_d, eef_rotm_d, eef_vel_d = self.get_eef_pose_vel(np.concatenate([self.adm_pose_ref[i], self.adm_vel_ref[i]]))
            eef_vel_d = self.adm_vel_ref[i]

            # process force
            world_force = np.zeros(6)
            force_limit = np.array([10,10,10,1,1,1])
            eef_force = self.force_sensor_data[i,0:3] - self.force_offset[i,:3]
            eef_torque = self.force_sensor_data[i,3:6] - self.force_offset[i,3:6]
            world_force[:3] = self.force_frame_offset @ eef_rotm @ eef_force
            world_force[3:] = self.force_frame_offset @ eef_rotm @ eef_torque
            world_force = np.clip(world_force, -force_limit, force_limit)

            # dynamics
            e = np.zeros(6)
            e[:3] = eef_pos - eef_pos_d
            eRd = eef_rotm @ eef_rotm_d.T
            dorn = trans_quat.mat2quat(eRd)
            do = dorn[1:]
            e[3:] = do

            e_dot = eef_vel - eef_vel_d
            MA = 1*world_force - np.multiply(self.adm_kp[i], e) - np.multiply(self.adm_kd[i], e_dot)
            adm_acc = np.divide(MA, self.adm_m[i])
            T = 1 / self.HZ
            adm_vel = eef_vel + adm_acc * T # This vel is for eef not link6, which we can control
            
            link6_pos, link6_rotm, link6_vel = self.get_link6_pose_vel_from_eef(eef_pos, eef_rotm, adm_vel)
            target_joint_vel[i] = np.linalg.pinv(self.jacobian[i]) @ link6_vel
            target_joint_vel[i] = np.clip(target_joint_vel[i], -np.array([0.4,0.4,0.4,0.5,0.5,0.5])/2, np.array([0.4,0.4,0.4,0.5,0.5,0.5])/2)
            if self.verbose:
                print("=="*50)
                print("robot",i)
                print("eef_pos",eef_pos)
                print("e",e)
                print("e_dot",e_dot)
                print("adm_vel",adm_vel)
                print("link6_vel",link6_vel)
                print("world_force",world_force)
        
        return target_joint_vel
    
    def set_joint_velocity(self, target_vel):
        T = 1 / self.HZ
        target_acc = np.zeros([self.num_robot,6])
        target_pos = np.zeros([self.num_robot,6])
        for i in range(self.num_robot):
            target_pos[i] = self.joint_pos[i] + T * (self.joint_vel[i] + target_vel[i]) / 2
            target_acc[i] = (target_vel[i] - self.joint_vel[i]) / T
        self.set_reference_traj(target_pos, target_vel, target_acc)

    def get_link6_pose_vel_from_eef(self, eef_pos, eef_rotm, eef_vel):
        link6_rotm = copy.copy(eef_rotm)
        link6_pos = eef_pos - link6_rotm @ self.eef_offset
        link6_vel = copy.copy(eef_vel)
        link6_vel[:3] = eef_vel[:3] - self.skew_symmetric(eef_vel[3:6]) @ link6_rotm @ self.eef_offset

        return link6_pos, link6_rotm, link6_vel

    def skew_symmetric(self,vec):
        return np.array([[0, -vec[2], vec[1]],
                         [vec[2], 0, -vec[0]],
                         [-vec[1], vec[0], 0]])
    
    def set_reference_traj(self, ref_joint, ref_vel, ref_acc):
        for i in range(self.num_robot):
            self.ref_joint[i] = np.concatenate([ref_joint[i], np.zeros(2)])
            self.ref_vel[i] = np.concatenate([ref_vel[i], np.zeros(2)])
            self.ref_acc[i] = np.concatenate([ref_acc[i], np.zeros(2)])
    
    def set_gripper(self, poses):
        assert poses.shape == (self.num_robot,)
        for i in range(self.num_robot):
            pose = poses[i]
            pose = 0.42 if pose > 0.42 else pose
            self.gripper_pose[i] = pose

    def sim_step(self):
        # apply computed torque control
        self.computed_torque_control()
        mujoco.mj_step(self.m, self.d)
        if self.render:
            self.viewer.sync()
        # get robot state
        self.get_pose_vel()
        self.update_force_sensor()
    
    def sim(self):
        curr_pose_vel = self.get_pose_vel()
        eef_pos, eef_rotm, eef_vel = self.get_eef_pose_vel(self.pose_vel[0])
        eef_pos_d = eef_pos + np.array([0.0,0.1,-0.1])
        eef_rotm_d = eef_rotm@trans_eul.euler2mat(0,0,0*np.pi/6)
        eef_vel_d = np.zeros(6)
        link6_pos, link6_rotm, link6_vel = self.get_link6_pose_vel_from_eef(eef_pos_d, eef_rotm_d,eef_vel_d)
        self.adm_pose_ref = np.zeros([self.num_robot,7])
        self.adm_pose_ref[0,:3] = link6_pos
        self.adm_pose_ref[0,3:7] = trans_quat.mat2quat(link6_rotm)
        self.adm_pose_ref[1,:3] = link6_pos + np.array([0.0,0.1,0.0])
        self.adm_pose_ref[1,3:7] = trans_quat.mat2quat(link6_rotm)
        start = time.time()
        while self.viewer.is_running() and time.time() - start < 3000:
            step_start = time.time()
            # self.ref_joint = np.array([[30.0,30.0,30.0,30.0,-90.0,30.0,0.42,0.42],
            #                            [40.0,40.0,40.0,40.0,-80.0,40.0,0.42,0.42]])/180*np.pi
            target_joint_vel = self.admittance_control()
            self.set_joint_velocity(target_joint_vel)
            self.sim_step()

            # Example modification of a viewer option: toggle contact points every two seconds.
            # with self.viewer.lock():
            #     self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.d.time % 2)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            self.viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

# register_env("mujoco_assembly-v0", Fanuc_mujoco_env)
if __name__ == '__main__':
    env = Fanuc_dual_arm_base()
    env.get_pose_vel()
    env.sim()
    # for i in range(5):
    #     env.reset()
    #     for _ in range(100):
    #         start = time.time()
    #         action = np.ones(12)
    #         action[:6] = np.array([-0.2,0,0,0,0,0])
    #         action[:2] = np.random.uniform(low=-0.1, high=0.1, size=2)
    #         ob, reward, done,_ = env.step(action)
    #         print(ob[:3])
    #         print(reward)
    #         print(time.time()-start)
    #         # time.sleep(0.1)