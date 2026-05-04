#!/bin/bash
#SBATCH --job-name=sac_sigent01
#SBATCH --partition=gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=train_sigma_ent_01.log
#SBATCH --error=train_sigma_ent_01.log

export PATH=/home/tw559/.conda/envs/reinflow/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tw559/.mujoco/mujoco210/bin:/usr/lib/nvidia
cd /home/tw559/steering_pi/KL_ReinFlow

export REINFLOW_DIR=/home/tw559/steering_pi/KL_ReinFlow
export REINFLOW_DATA_DIR=/home/tw559/steering_pi/KL_ReinFlow/hf_cache/data-offline
export REINFLOW_LOG_DIR=/home/tw559/steering_pi/KL_ReinFlow/log
export REINFLOW_WANDB_ENTITY=terrywang330-cornell-university
export D4RL_SUPPRESS_IMPORT_ERROR=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=online

python script/run.py --config-name=ft_sac_residual_flow_mlp --config-path=../cfg/gym/finetune/hopper-v2 \
  base_policy_path=hf_cache/log/log_gym_d4rl_pretrained/hopper-medium-v2/1-ReFlow/state_40.pt \
  normalization_path=hf_cache/data-offline/gym_d4rl/hopper-medium-v2/normalization.npz \
  train.n_train_itr=200000 \
  env.n_envs=40 \
  train.sigma_entropy_weight=0.1
