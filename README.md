# Variable admittance control simulation of Fanuc LRMate 200iD robot

## Installation
### Step 1: Install mujoco (the newest version >=3.0.0). PLease follow the documentations.
### Step 2: Create an conda env:
```
conda create --name mujoco3 python=3.9 pip
conda activate mujoco3
pip install -r requirements.txt
```
### Step 3: Install torch & Rllib (with suitable cuda version)

## Training & evaluate
### Training (for peg-in-hole):
```
python ray_scripts/ray_ppo_test.py
```
### Evaluate (for pivoting)
```
python ray_scripts/ray_ppo_pivoting_evaluate.py
```