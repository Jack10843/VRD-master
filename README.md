# Code for VRD (D4RL demos)

Official implementation for Value-Rectified Distillation for Flow-based Offline Reinforcement Learning.

## Environment

1. Install [MuJoCo version 2.0](https://www.roboti.us/download.html) at ~/.mujoco/mujoco200 and copy license key to ~/.mujoco/mjkey.txt
2. Create a conda environment
```
conda env create -f conda_env.yml
conda activate vrd
```
3. Install [D4RL](https://github.com/Farama-Foundation/D4RL/)

## Usage

### Offline RL

Run the following command to train offline RL on D4RL.

```
python main.py --config configs/offline/halfcheetah-medium-replay.yml
python main.py --config configs/offline/antmaze-medium-diverse.yml
```

#### Logging

This codebase uses tensorboard. You can view saved runs with:

```
tensorboard --logdir <run_dir>
```

## Codes for Minari and OGBench will be released soon.
