from typing import Optional, Tuple
from Fanuc_mujoco_dual_arm_base import Fanuc_dual_arm_base
import numpy as np
# from gym import spaces
import gymnasium
from gymnasium import spaces,register
import transforms3d.quaternions as trans_quat
import transforms3d.euler as trans_eul
from source.lrmate_kine_base import IK
import mujoco
import time

class Fanuc_dual_arm_pivoting(Fanuc_dual_arm_base):
    def __init__(self):
        super().__init__(render=True, verbose=False)
        # reset the two arms to the initial position

        # a lot of RL settings
        self.work_space_origin = np.array([0.5, 0.6, 0.0])
        self.work_space_origin_rotm = np.array([[0, 1, 0],
                                                [0, 0, -1],
                                                [-1, 0, 0]])
        self.object_goal_pos = np.array([0.5, 0.6, 0.0])
        self.object_goal_rotm = np.array([[1, 0, 0],
                                          [0, 0, -1],
                                          [0, 1, 0]])
        self.goal_threshold = 5.0 / 180 * np.pi
        self.force_noise = True
        self.force_noise_level = 0.2
        self.force_limit = 20
        self.moving_pos_threshold = 2.5
        self.moving_ori_threshold = 4

        # action space and observation space
        # action space: eef desired velocity, and admittance gains (kp only) for both arms
        # observation space: eef position, orientation, and force for both arms, combine with the object position and orientation
        self.action_vel_high = 0.1 * np.ones(6)
        self.action_vel_low = -0.1 * np.ones(6)
        self.action_kp_high = 200 * np.ones(6)
        self.action_kp_low = 100 * np.ones(6)
        self.action_space = spaces.Box(low=-np.ones(12), high=np.ones(12), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1., high=1., shape=self.get_RL_obs().shape, dtype=np.float32)

        # reset
        self.reset()
        self.force_calibration()


    
    def get_RL_obs(self):
        # get the observation for RL
        # observation: eef position, orientation, and force for both arms, combine with the object position and orientation
        state = []
        for i in range(self.num_robot):
            eef_pos, eef_world_rotm, eef_vel = self.get_eef_pose_vel(self.pose_vel[i])
            eef_pos = eef_pos - self.work_space_origin
            eef_rotm = self.work_space_origin_rotm.T@eef_world_rotm
            eef_eul = trans_eul.mat2euler(eef_rotm)
            world_force = np.zeros(6)
            eef_force = self.force_sensor_data[i] - self.force_offset[i]
            world_force[:3] = self.force_frame_offset @ eef_world_rotm @ eef_force[:3]
            world_force[3:6] = self.force_frame_offset @ eef_world_rotm @ eef_force[3:6]
            if self.force_noise:
                world_force = world_force + np.random.normal(0, self.force_noise_level, 6)
            world_force = np.clip(world_force, -10, 10)
            state.append(np.concatenate([eef_pos, eef_eul, eef_vel, world_force]))
        # obtain object position and orientation
        obj_pos, obj_rot_mat, obj_vel, _ = self.get_gemo_pose_vel("object_geom_1")
        obj_pos = obj_pos - self.work_space_origin
        obj_eul = trans_eul.mat2euler(obj_rot_mat)
        state.append(np.concatenate([obj_pos, obj_eul]))
        state = np.concatenate(state)
        # TODO: normalize the observation
        
        return state

    def reset(self):
        # reset the two arms to the initial position which is close to object
        robot_1_pose = np.array([0.5, 0.4-0.15, 0.27, 0.0, np.pi, np.pi/2])
        robot_2_pose = np.array([0.5, (0.4+0.15) - self.robot_y_offset, 0.27, 0.0, np.pi, np.pi/2])
        init_j_pose = IK(robot_1_pose)
        self.d.qpos[:6] = init_j_pose
        init_j_pose = IK(robot_2_pose)
        self.d.qpos[8:8+6] = init_j_pose
        # set object position and orientation
        self.d.qpos[16:] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        mujoco.mj_step(self.m, self.d)
        # TODO: add some random noise to the initial position

        return self.get_RL_obs()
    
    def process_action(self, action):
        # Normalize actions
        desired_vel = np.clip(action[:6], -1, 1)
        desired_kp = np.clip(action[6:12], -1, 1)
        desired_vel = (self.action_vel_high + self.action_vel_low)/2 + np.multiply(desired_vel, (self.action_vel_high - self.action_vel_low)/2)
        desired_kp = (self.action_kp_high + self.action_kp_low)/2 + np.multiply(desired_kp, (self.action_kp_high - self.action_kp_low)/2)
        return desired_vel, desired_kp
    
    def step(self, action):
        # action: eef desired velocity, and admittance gains (kp only) for both arms
        # action: [vel_1, kp_1, vel_2, kp_2]
        # vel: [x, y, z, roll, pitch, yaw]
        # kp: [x, y, z, roll, pitch, yaw]
        # vel: [-0.1, 0.1]
        # kp: [100, 200]

        # process action to desired velocity and kp
        desired_vel = np.zeros((self.num_robot, 6))
        for i in range(self.num_robot):
            desired_vel[i], desired_kp = self.process_action(action[i*12:(i+1)*12])
            self.adm_kp[i] = desired_kp
            self.adm_kd[i] = 2 * np.sqrt(np.multiply(self.adm_kp[i], self.adm_m[i]))
        
        init_ob = self.get_RL_obs()
        for i in range(50):
            current_ob = self.get_RL_obs()
            terminal_flag = [False, False]
            for j in range(self.num_robot):
                # first, check the moving threshold and the force threshold for both arms
                # if one of these two conditions is satisfied, then stop that arm
                # if both arms stop, then stop the simulation

                robot_pos = current_ob[j*18:j*18+6]
                robot_force = current_ob[j*18+12:j*18+18]
                if np.abs(np.dot(robot_force, desired_vel[j]) / np.linalg.norm(desired_vel[j] + 1e-6, ord=2)) > self.force_limit:
                    terminal_flag[j] = True
                    break
                delta_ob = robot_pos - init_ob[j*18:j*18+6]
                # Moving threshold
                if np.linalg.norm(delta_ob[0:3], ord=2) > self.moving_pos_threshold or np.linalg.norm(delta_ob[3:6], ord=2)\
                        > self.moving_ori_threshold / 180 * np.pi:
                    terminal_flag[j] = True
                    break
                if self.check_ori_dist_2_goal(current_ob[-3:]) < self.goal_threshold:
                    desired_vel = np.zeros([self.num_robot, 6])
                    terminal_flag[j] = True
                
                done = False
                eef_pos, eef_rotm, eef_vel = self.get_eef_pose_vel(self.pose_vel[j])
                link6_pos, link6_rotm, link6_vel = self.get_link6_pose_vel_from_eef(eef_pos, eef_rotm, desired_vel[j])
                self.adm_pose_ref[j,:3] = link6_pos
                self.adm_pose_ref[j,3:7] = trans_quat.mat2quat(link6_rotm)
                self.adm_vel_ref[j] = link6_vel
            if terminal_flag[0] and terminal_flag[1]:
                break
            target_joint_vel = self.admittance_control()
            self.set_joint_velocity(target_joint_vel)
            self.sim_step()

        self.clear_motion()
        ob = self.get_RL_obs()
        # evalute reward
        dist = self.check_ori_dist_2_goal(ob[-3:])
        # print(dist)
        reward = np.pi / 2 - dist
        if reward < 0:
            reward = 0
        if dist < self.goal_threshold:
            done = True
        return ob, reward, done, dict(reward_dist=reward)
    
    def clear_motion(self):
        # clear robot motion (make robot stay still)
        desired_vel = np.zeros([self.num_robot, 6])
        for _ in range(10):
            self.adm_pose_ref = self.pose_vel[:, :7]
            self.adm_vel_ref = desired_vel
            target_joint_vel = self.admittance_control()
            self.set_joint_velocity(target_joint_vel)
            self.sim_step()
    
    def check_ori_dist_2_goal(self, eul):
        # rotm = trans_quat.quat2mat(quat)
        rotm = trans_eul.euler2mat(eul[0], eul[1], eul[2], axes='sxyz')
        R = rotm.T @ self.object_goal_rotm
        return np.arccos((np.trace(R) - 1) / 2.0)

if __name__ == "__main__":
    env = Fanuc_dual_arm_pivoting()
    # env.clear_motion()
    # env.sim()
    for _ in range(10):
        env.reset()
        for i in range(100):
            print(i)
            action = np.array([0,1,0,1,0,0,1,1,1,1,1,1,
                               0,-1,0.2,-1,0,0,1,1,1,1,1,1])
            ob, reward, done,_ = env.step(action)
            print(reward)
            # time.sleep(1)