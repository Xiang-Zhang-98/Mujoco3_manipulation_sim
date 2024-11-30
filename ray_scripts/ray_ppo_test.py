import sys
import os
sys.path.insert(0, os.getcwd())
from envs.ray_mujoco_insertion import Fanuc_mujoco_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("mujoco_assembly-v0")
    .rollouts(num_rollout_workers=12,rollout_fragment_length=30)
    .framework("torch")
    .training(model={"fcnet_hiddens": [128, 128]},train_batch_size=2000)
    .evaluation(evaluation_num_workers=1)
)

trail_name = "PPO#1"

algo = config.build()  # 2. build the algorithm,

for i in range(500):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save(checkpoint_dir=os.getcwd()+"/ray_result/"+trail_name).checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")

algo.evaluate()  # 4. and evaluate it.