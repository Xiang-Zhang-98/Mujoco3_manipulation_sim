from ray.rllib.algorithms.algorithm import Algorithm
import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())
import gymnasium as gym
from envs.ray_mujoco_dual_pivoting import Fanuc_dual_arm_pivoting
from ray.rllib.algorithms.ppo import PPOConfig
import time


trail_name = "PPO_dual_pivoting_-dist_w_rot_lim_w_robot_dist_f0.3_ws_H40_final_reward_100_trail2"
path_to_checkpoint = os.getcwd()+"/ray_result/"+trail_name
config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("mujoco_dual_pivoting-v0")
    .rollouts(num_rollout_workers=1,rollout_fragment_length=40)
    .framework("torch")
    .training(model={"fcnet_hiddens": [256, 256]},train_batch_size=2000)
    .evaluation(evaluation_num_workers=1)
)

algo = config.build()  # 2. build the algorithm,
# algo = Algorithm.from_checkpoint(path_to_checkpoint)
algo.restore(path_to_checkpoint)
env = gym.make("mujoco_dual_pivoting-v0",render=True)

for i in range(200):
    obs,_ = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    while not terminated and not truncated:
        action = algo.compute_single_action(obs)
        # action = np.array([0,0,0,0,0,1,1,1,1,1,1,1,
        #                    0,0,0,0,0,0,1,1,1,1,1,1])
        obs, reward, terminated, truncated, info = env.step(action)
        # time.sleep(2)
        # print(obs[-6:-3])
        episode_reward += reward
    print(f"Episode reward: {episode_reward}")
