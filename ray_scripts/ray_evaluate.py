from ray.rllib.algorithms.algorithm import Algorithm
import os
import sys
sys.path.insert(0, os.getcwd())
import gymnasium as gym
from envs.ray_mujoco_insertion import Fanuc_mujoco_env

trail_name = "PPO#1"
path_to_checkpoint = os.getcwd()+"/ray_result/"+trail_name
algo = Algorithm.from_checkpoint(path_to_checkpoint)
env = gym.make("mujoco_assembly_ray-v0",render=True)

for i in range(200):
    obs,_ = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    while not terminated and not truncated:
        action = algo.compute_single_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
    print(f"Episode reward: {episode_reward}")
