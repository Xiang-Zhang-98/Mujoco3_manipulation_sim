import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import gym
from gym import spaces,register
import transforms3d.quaternions as trans_quat
import transforms3d.euler as trans_eul
import copy
from .source.lrmate_kine_base import IK
from ray.tune.registry import register_env

class Fanuc_mujoco_env(gym.Env):
    def __init__(self, render=True):
        super(Fanuc_mujoco_env, self).__init__()
        self.m = mujoco.MjModel.from_xml_path('envs/source/LRMate_200iD.xml')
        self.d = mujoco.MjData(self.m)
        # joint control gains & variables
        self.joint_kp = np.array([100, 100, 100, 100, 100, 100, 100, 100])
        self.joint_kd = 2 * np.sqrt(self.joint_kp)
        self.joint_pos = np.zeros(6)
        self.joint_vel = np.zeros(6)
        self.joint_acc = np.zeros(6)
        self.ref_joint = np.zeros(8)
        self.ref_vel = np.zeros(8)
        self.ref_acc = np.zeros(8)
        self.gripper_pose = 0.031
        self.set_gripper(self.gripper_pose)

        # admittance control gains
        self.adm_kp = 10 * np.array([1, 1, 1, 1, 1, 1])
        self.adm_m = 1 * np.array([1, 1, 1, 1, 1, 1])
        self.adm_kd = 4 * np.sqrt(np.multiply(self.adm_kp, self.adm_m))
        self.adm_pose_ref = np.zeros(7)
        self.adm_vel_ref = np.zeros(6)
        self.HZ = 125

        # place holder for robot state
        self.pose_vel = np.zeros(13)
        self.force_sensor_data = np.zeros(6)
        self.force_offset = np.zeros(6)
        self.jacobian = np.zeros((6, 6))

        # peg-in-hole task setting
        self.eef_offset = np.array([0.3-0.0341, 0.0, 0.0])
        self.work_space_xy_limit = 4
        self.work_space_z_limit = 8
        self.work_space_rollpitch_limit = np.pi * 5 / 180.0
        self.work_space_yaw_limit = np.pi * 10 / 180.0
        self.work_space_origin = np.array([0.50, 0, 0.1])
        self.work_space_origin_rotm = np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]])
        self.force_frame_offset = np.array([[-1, 0, 0],
                                            [0, -1, 0],
                                            [0, 0, -1]])
        self.goal = np.array([0, 0, 0])
        self.goal_ori = np.array([0, 0, 0])
        self.noise_level = 0.2
        self.ori_noise_level = 0.5
        self.use_noisy_state = False
        self.state_offset = np.zeros(18)
        self.force_noise = True
        self.force_noise_level = 0.2
        self.force_limit = 20
        self.evaluation = True #self.Render
        self.moving_pos_threshold = 2.5
        self.moving_ori_threshold = 4

        # RL setting
        self.obs_high = [self.work_space_xy_limit,self.work_space_xy_limit,self.work_space_z_limit,
                         self.work_space_rollpitch_limit,self.work_space_rollpitch_limit,self.work_space_yaw_limit,
                         0.1,0.1,0.1,0.1,0.1,0.1,
                         10,10,10,10,10,10]
        self.obs_high = np.array(self.obs_high)
        self.observation_space = spaces.Box(low=-1., high=1., shape=self.get_RL_obs().shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-np.ones(12), high=np.ones(12), dtype=np.float32)
        self.action_vel_high = 0.1 * np.array([1,1,0.5,1,1,1])
        self.action_vel_low = -0.1 * np.ones(6)
        self.action_kp_high = 200 * np.array([1,1,1,1,1,1])
        self.action_kp_low = 1 * np.array([1,1,1,1,1,1])
        
        # init
        self.render = render
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        self.reset()
        mujoco.mj_step(self.m, self.d)
        # get robot state
        self.get_pose_vel()
        self.get_force_sensor_data()
        self.force_calibration()
    
    def get_RL_obs(self):
        eef_pos, eef_world_rotm, eef_vel = self.get_eef_pose_vel(self.pose_vel)
        eef_pos = eef_pos - self.work_space_origin
        eef_rotm = eef_world_rotm @ self.work_space_origin_rotm.T
        eef_eul = trans_eul.mat2euler(eef_rotm)
        world_force = np.zeros(6)
        eef_force = self.force_sensor_data - self.force_offset
        world_force[:3] = self.force_frame_offset @ eef_world_rotm @ eef_force[:3]
        world_force[3:6] = self.force_frame_offset @ eef_world_rotm @ eef_force[3:6]
        if self.force_noise:
            world_force = world_force + np.random.normal(0, self.force_noise_level, 6)
        world_force = np.clip(world_force, -10, 10)
        state = np.concatenate([100*eef_pos, eef_eul, eef_vel, world_force])
        # state = np.clip(state, -self.obs_high, self.obs_high)
        if self.use_noisy_state:
            return state + self.state_offset
        else:
            return state
    
    def step(self, action):
        # step function for RL
        desired_vel, desired_kp = self.process_action(action)
        self.adm_kp = desired_kp
        self.adm_kd = 2 * np.sqrt(np.multiply(self.adm_kp, self.adm_m))
        init_ob = self.get_RL_obs()
        # keep the same action for a short time
        for i in range(50):
            ob = self.get_RL_obs()
            curr_force = ob[12:]
            off_work_space = False
            # Force limit constraint
            if np.abs(np.dot(curr_force, desired_vel) / np.linalg.norm(desired_vel + 1e-6, ord=2)) > self.force_limit:
                break
            delta_ob = ob - init_ob
            # Moving threshold
            if np.linalg.norm(delta_ob[0:3], ord=2) > self.moving_pos_threshold or np.linalg.norm(delta_ob[3:6], ord=2)\
                    > self.moving_ori_threshold / 180 * np.pi:
                break
            # check workspace, if out of workspace, then reverse the velocity
            # if np.abs(ob[0]) > self.work_space_xy_limit:
            #     desired_vel[0] = -2*self.action_vel_high[0] * np.sign(ob[0])
            #     off_work_space = True
            # if np.abs(ob[1]) > self.work_space_xy_limit:
            #     desired_vel[1] = -2*self.action_vel_high[1] * np.sign(ob[1])
            #     off_work_space = True
            # if np.abs(ob[3]) > self.work_space_rollpitch_limit:
            #     desired_vel[3] = -2*self.action_vel_high[3] * np.sign(ob[3])
            #     off_work_space = True
            # if np.abs(ob[4]) > self.work_space_rollpitch_limit:
            #     desired_vel[4] = -2*self.action_vel_high[4] * np.sign(ob[4])
            #     off_work_space = True
            # if np.abs(ob[5]) > self.work_space_yaw_limit:
            #     desired_vel[5] = -2*self.action_vel_high[5] * np.sign(ob[5])
            #     off_work_space = True
            # if ob[2] > self.work_space_z_limit:
            #     desired_vel[2] = -self.action_vel_high[2]
            #     off_work_space = True
            # print(desired_vel)
            # check done
            if np.linalg.norm(ob[0:3] - self.goal) < 0.3:
                done = False
                desired_vel = np.zeros(6)  # if reach to goal, then stay
                self.adm_pose_ref = self.pose_vel[:7]
                self.adm_vel_ref = desired_vel
                target_joint_vel = self.Cartersian_vel_control(desired_vel)
            # if out of workspace, then use vel control to pull back
            elif off_work_space:
                done = False
                eef_pos, eef_rotm, eef_vel = self.get_eef_pose_vel(self.pose_vel)
                link6_pos, link6_rotm, link6_vel = self.get_link6_pose_vel_from_eef(eef_pos, eef_rotm, desired_vel)
                self.adm_pose_ref[:3] = link6_pos
                self.adm_pose_ref[3:7] = trans_quat.mat2quat(link6_rotm)
                self.adm_vel_ref = link6_vel
                # self.adm_pose_ref = self.pose_vel[:7]
                # self.adm_vel_ref = desired_vel
                target_joint_vel = self.Cartersian_vel_control(desired_vel)
            else:
                done = False
                eef_pos, eef_rotm, eef_vel = self.get_eef_pose_vel(self.pose_vel)
                link6_pos, link6_rotm, link6_vel = self.get_link6_pose_vel_from_eef(eef_pos, eef_rotm, desired_vel)
                self.adm_pose_ref[:3] = link6_pos
                self.adm_pose_ref[3:7] = trans_quat.mat2quat(link6_rotm)
                self.adm_vel_ref = link6_vel
                # self.adm_pose_ref = self.pose_vel[:7]
                # self.adm_vel_ref = desired_vel
                target_joint_vel = self.admittance_control()

            self.set_joint_velocity(target_joint_vel)
            self.sim_step()

        ob = self.get_RL_obs()
        # evalute reward
        dist = np.linalg.norm(ob[0:3] - self.goal)
        # if dist < 0.3:
        #     done = False
        #     reward = 10
        # else:
        #     done = False
        #     reward = np.power(10, 1 - dist)
        reward = -dist
        if self.evaluation and np.linalg.norm(ob[2] - self.goal[2]) < 0.3:
            done = True
        return ob, reward, done, dict(reward_dist=reward)
    
    def process_action(self, action):
        # Normalize actions
        desired_vel = np.clip(action[:6], -1, 1)
        desired_kp = np.clip(action[6:12], -1, 1)
        desired_vel = (self.action_vel_high + self.action_vel_low)/2 + np.multiply(desired_vel, (self.action_vel_high - self.action_vel_low)/2)
        desired_kp = (self.action_kp_high + self.action_kp_low)/2 + np.multiply(desired_kp, (self.action_kp_high - self.action_kp_low)/2)
        return desired_vel, desired_kp

    def reset(self):
        # set eef init pose
        init_c_pose = np.array([0.5, 0.0, 0.40, 0.0, np.pi, np.pi])
        l = np.array([3,3,0.5])/100
        cube = np.random.uniform(low=-l, high=l)
        init_c_pose[0:3] = init_c_pose[0:3] + cube
        init_j_pose = IK(init_c_pose)
        self.d.qpos[:6] = init_j_pose
        self.force_calibration()
        # Domain-randomization
        self.state_offset[:2] = np.random.normal(0, np.array([self.noise_level,self.noise_level]))
        angle = np.random.normal(0, self.ori_noise_level / 180 * np.pi)
        self.state_offset[5] = angle
        return self.get_RL_obs()

    def force_calibration(self, H=100):
        """
        Calibrate force sensor reading
        H: force history horizon
        """
        force_history = np.zeros([H, 6])
        self.sim_step()
        self.set_reference_traj(
            self.joint_pos, 0 * self.joint_vel, 0 * self.joint_acc
        )
        for _ in range(H):
            self.sim_step()
            force_history[_, :] = self.force_sensor_data
        self.force_offset = np.mean(force_history[int(H / 2):], axis=0)
    
    def get_pose_vel(self):
        # get robot end-effector pose and velocity
        geom_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, 'link6')

        # Get position and rotation matrix of the end-effector (Link 6)
        link6_pos = self.d.geom_xpos[geom_id]
        link6_rot_mat = self.d.geom_xmat[geom_id].reshape([3,3])@np.array([[ 1.0000,  0.0008,  0.0003],
                                                            [-0.0008,  0.8373,  0.5467],
                                                            [ 0.0002, -0.5467,  0.8373]]).T

        # Convert rotation matrix to quaternion
        # link6_rot_quat = np.zeros(4)
        # mujoco.mju_mat2Quat(link6_rot_quat,link6_rot_mat)
        link6_rot_quat = trans_quat.mat2quat(link6_rot_mat)
        
        # Calculate Jacobian
        jacp = np.zeros([3, self.m.nv])
        jacr = np.zeros([3, self.m.nv])
        mujoco.mj_jacGeom(self.m, self.d, jacp, jacr, geom_id)
        Full_jacobian = np.concatenate([jacp, jacr])

        # Calculate Cartesian velocity of Link 6
        link6_vel = np.dot(Full_jacobian, self.d.qvel[:8])  # Assuming 8 DoFs

        # Combine position, orientation (quaternion), and velocity
        pose_vel = np.concatenate([link6_pos, link6_rot_quat, link6_vel[:6]])

        # update robot state
        self.jacobian = Full_jacobian[:6, :6]
        self.pose_vel = pose_vel
        self.joint_pos = self.d.qpos[:6].copy()
        self.joint_vel = self.d.qvel[:6].copy()

        return pose_vel
    
    def get_force_sensor_data(self):
        # get force sensor data
        sensor_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "force_ee")

        # Get address and dimension of the sensor
        adr = self.m.sensor_adr[sensor_id]
        dim = self.m.sensor_dim[sensor_id]
        force = np.copy(self.d.sensordata[adr:adr + dim])
        # get torque sensor data
        sensor_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "torque_ee")
        adr = self.m.sensor_adr[sensor_id]
        dim = self.m.sensor_dim[sensor_id]
        torque = np.copy(self.d.sensordata[adr:adr + dim])
        force_torque = np.concatenate([force, torque])
        # update robot state
        self.force_sensor_data = force_torque

        return force_torque
    
    def computed_torque_control(self):
        # Assuming self.m and self.d are equivalent to m and d in the C++ code

        # Compute inverse dynamics forces
        mujoco.mj_rne(self.m, self.d, 0, self.d.qfrc_inverse)
        for i in range(self.m.nv):
            self.d.qfrc_inverse[i] += (self.m.dof_armature[i] * self.d.qacc[i] -
                                    self.d.qfrc_passive[i] -
                                    self.d.qfrc_constraint[i])

        # Error and error derivative
        e = self.d.qpos[:8] - self.ref_joint  # ref_joint should be defined or passed as an argument
        e_dot = self.d.qvel[:8] - self.ref_vel  # ref_vel should be defined or passed as an argument

        # Control law components
        kve_dot = np.multiply(self.joint_kd, e_dot)  # kv should be defined or passed as an argument
        kpe = np.multiply(self.joint_kp, e)  # kp should be defined or passed as an argument
        inertial_pd = self.ref_acc - kve_dot - kpe  # ref_acc should be defined or passed as an argument

        # Compute full inertia matrix
        M = np.zeros((self.m.nv, self.m.nv))
        mujoco.mj_fullM(self.m, M, self.d.qM)
        M_robot = M[:self.m.nu, :self.m.nu]  # Assuming self.m.nu is the number of actuators

        # Compute the control torques
        inertial_torque = np.dot(M_robot, inertial_pd)

        # Apply control and inverse dynamics torques
        self.d.ctrl[:8] = np.clip(inertial_torque + self.d.qfrc_inverse[:8],-5,5)

        # Set gripper control
        self.d.ctrl[6] = self.gripper_pose  # gripper_pose should be defined or passed as an argument
        self.d.ctrl[7] = self.gripper_pose
    
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
        # print("=="*50)
        # print("cartesian")
        return target_joint_vel

    def admittance_control_link6(self):
        link6_pos = self.pose_vel[:3]
        link6_rotm = trans_quat.quat2mat(self.pose_vel[3:7])
        link6_vel = self.pose_vel[7:]
        link6_pos_d = self.adm_pose_ref[:3]
        link6_rotm_d = trans_quat.quat2mat(self.adm_pose_ref[3:7])
        link6_vel_d = self.adm_vel_ref

        # process force
        world_force = np.zeros(6)
        force_limit = np.array([10,10,10,1,1,1])
        eef_force = self.force_sensor_data[0:3] - self.force_offset[:3]
        eef_torque = self.force_sensor_data[3:6] - self.force_offset[3:6]
        world_force[:3] = self.force_frame_offset @ eef_rotm @ eef_force
        world_force[3:] = self.force_frame_offset @ eef_rotm @ eef_torque
        world_force = np.clip(world_force, -force_limit, force_limit)

        e = np.zeros(6)
        e[:3] = link6_pos - link6_pos_d
        eRd = link6_rotm @ link6_rotm_d.T
        dorn = trans_quat.mat2quat(eRd)
        do = dorn[1:]
        e[3:] = do

        e_dot = link6_vel - link6_vel_d
        MA = 1*world_force - np.multiply(self.adm_kp, e) - np.multiply(self.adm_kd, e_dot)
        adm_acc = np.divide(MA, self.adm_m)
        T = 1 / self.HZ
        adm_vel = link6_vel + adm_acc * T # This vel is for eef not link6, which we can control
        target_joint_vel = np.linalg.pinv(self.jacobian) @ adm_vel
        return target_joint_vel

    def admittance_control(self, ctl_ori=True):
        ## Get robot motion from desired dynamics
        eef_pos, eef_rotm, eef_vel = self.get_eef_pose_vel(self.pose_vel)
        eef_pos_d, eef_rotm_d, eef_vel_d = self.get_eef_pose_vel(np.concatenate([self.adm_pose_ref, self.adm_vel_ref]))
        eef_vel_d = self.adm_vel_ref

        # process force
        world_force = np.zeros(6)
        force_limit = np.array([10,10,10,1,1,1])
        eef_force = self.force_sensor_data[0:3] - self.force_offset[:3]
        eef_torque = self.force_sensor_data[3:6] - self.force_offset[3:6]
        world_force[:3] = self.force_frame_offset @ eef_rotm @ eef_force
        world_force[3:] = self.force_frame_offset @ eef_rotm @ eef_torque
        world_force = np.clip(world_force, -force_limit, force_limit)

        # dynamics
        e = np.zeros(6)
        e[:3] = eef_pos - eef_pos_d
        if ctl_ori:
            eRd = eef_rotm @ eef_rotm_d.T
            dorn = trans_quat.mat2quat(eRd)
            do = dorn[1:]
            e[3:] = do

        e_dot = eef_vel - eef_vel_d
        MA = 1*world_force - np.multiply(self.adm_kp, e) - np.multiply(self.adm_kd, e_dot)
        adm_acc = np.divide(MA, self.adm_m)
        T = 1 / self.HZ
        adm_vel = eef_vel + adm_acc * T # This vel is for eef not link6, which we can control
        if not ctl_ori:
            adm_vel[3:] = 0 * adm_vel[3:]
        
        link6_pos, link6_rotm, link6_vel = self.get_link6_pose_vel_from_eef(eef_pos, eef_rotm, adm_vel)
        target_joint_vel = np.linalg.pinv(self.jacobian) @ link6_vel
        target_joint_vel = np.clip(target_joint_vel, -np.array([0.4,0.4,0.4,0.5,0.5,0.5])/2, np.array([0.4,0.4,0.4,0.5,0.5,0.5])/2)
        # print("=="*50)
        # print("e",e)
        # print("e_dot",e_dot)
        # print("adm_vel",adm_vel)
        # print("link6_vel",link6_vel)
        # print("world_force",world_force)
        # print("control",self.d.ctrl[:8])
        return target_joint_vel
    
    def set_joint_velocity(self, target_vel):
        T = 1 / self.HZ
        target_pos = self.joint_pos + T * (self.joint_vel + target_vel) / 2
        target_acc = (target_vel - self.joint_vel) / T
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
        assert (
                ref_joint.shape == (6,) and ref_vel.shape == (6,) and ref_acc.shape == (6,)
        )
        self.ref_joint = np.concatenate([ref_joint, np.zeros(2)])
        self.ref_vel = np.concatenate([ref_vel, np.zeros(2)])
        self.ref_acc = np.concatenate([ref_acc, np.zeros(2)])
    
    def set_gripper(self, pose):
        pose = 0.42 if pose > 0.42 else pose
        self.gripper_pose = pose

    def sim_step(self):
        # apply computed torque control
        self.computed_torque_control()
        mujoco.mj_step(self.m, self.d)
        if self.render:
            self.viewer.sync()
        # get robot state
        self.get_pose_vel()
        self.get_force_sensor_data()
    
    def sim(self):
        # set ref traj
        curr_pose_vel = self.get_pose_vel()
        eef_pos, eef_rotm, eef_vel = self.get_eef_pose_vel(self.pose_vel)
        eef_pos_d = eef_pos + np.array([0.0,0.0,0.1])
        eef_rotm_d = eef_rotm@trans_eul.euler2mat(0,0,1*np.pi/6)
        eef_vel_d = np.zeros(6)
        link6_pos, link6_rotm, link6_vel = self.get_link6_pose_vel_from_eef(eef_pos_d, eef_rotm_d,eef_vel_d)
        self.adm_pose_ref = np.zeros(7)
        self.adm_pose_ref[:3] = link6_pos
        self.adm_pose_ref[3:7] = trans_quat.mat2quat(link6_rotm)
        # self.adm_pose_ref[:3] = curr_pose_vel[:3]+np.array([0.0,0.0,0.0])
        # delta_ori = trans_quat.mat2quat(trans_quat.quat2mat(curr_pose_vel[3:7])@trans_eul.euler2mat(0,0,1*np.pi/6))
        # self.adm_pose_ref[3:7] = delta_ori
        # self.adm_pose_ref = np.array([ 0.4899,  0.0352,  0.4076,  1, 0,  0,  0])
        # print(self.adm_pose_ref)
        start = time.time()
        while self.viewer.is_running() and time.time() - start < 3000:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            # eef_pos, eef_rotm, eef_vel = self.get_eef_pose_vel(self.pose_vel)
            # link6_pos, link6_rotm, link6_vel = self.get_link6_pose_vel_from_eef(eef_pos, eef_rotm, 
            #                                                                     np.array([0.00,0,0,0.0,0.0,0.00]))
            target_joint_vel = self.admittance_control(ctl_ori=True)
            # target_joint_vel = self.Cartersian_vel_control(link6_vel)
            print(self.pose_vel[:3])
            print(trans_quat.quat2mat(self.pose_vel[3:7]))
            self.set_joint_velocity(target_joint_vel)
            # self.ref_joint = np.array([0, 0, 0, 0, -90, 0, 0.42, 0.42])/180*np.pi
            self.sim_step()
            # time.sleep(0.01)
            # print(self.d.time)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with self.viewer.lock():
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.d.time % 2)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            self.viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
register(
    id="mujoco_assembly-v0",
    entry_point=Fanuc_mujoco_env,
)
# register_env("mujoco_assembly-v0", Fanuc_mujoco_env)
if __name__ == '__main__':
    env = Fanuc_mujoco_env()
    # env.sim()
    for i in range(5):
        env.reset()
        for _ in range(100):
            start = time.time()
            action = np.ones(12)
            action[:6] = np.array([-0.2,0,0,0,0,0])
            action[:2] = np.random.uniform(low=-0.1, high=0.1, size=2)
            ob, reward, done,_ = env.step(action)
            print(ob[:3])
            print(reward)
            print(time.time()-start)
            # time.sleep(0.1)