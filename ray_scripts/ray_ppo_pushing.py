import sys
import os
sys.path.insert(0, os.getcwd())
from envs.ray_mujoco_dual_pivoting import Fanuc_dual_arm_pivoting
# from envs.ray_mujoco_insertion import Fanuc_mujoco_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
import gymnasium as gym
from gymnasium import spaces,register
from ray.tune.registry import register_env

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("mujoco_dual_pivoting-v0")
    .rollouts(num_rollout_workers=24,rollout_fragment_length=40)
    .framework("torch")
    .training(model={"fcnet_hiddens": [256, 256]},train_batch_size=4800)
    .evaluation(evaluation_num_workers=1)
)
warm_start_trail = None
trail_name = "PPO_dual_pushing_just_dis_reward"

algo = config.build()  # 2. build the algorithm,

if warm_start_trail:
    checkpoint_dir = os.getcwd()+"/ray_result/"+warm_start_trail
    algo.restore(checkpoint_dir)

for i in range(500):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save(checkpoint_dir=os.getcwd()+"/ray_result/"+trail_name).checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")

algo.evaluate()  # 4. and evaluate it.