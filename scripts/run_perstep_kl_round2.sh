#!/bin/bash
# Round 2: Higher KL weight + EMA absorption experiments.
# Run 5b: perstep_exact, kl_weight=0.5 (10x previous)
# Run 7:  perstep_exact + EMA absorption (kl_weight=0.05, ema_absorb_freq=100, tau=0.01)
# Run 8:  perstep_hutchinson + EMA absorption (same settings)
#
# Launch: srun --jobid=<JOBID> --overlap bash scripts/run_perstep_kl_round2.sh

set -euo pipefail

cd /home/tw559/steering_pi/KL_ReinFlow
mkdir -p slurm_logs log

export PATH=/home/tw559/.conda/envs/reinflow/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/home/tw559/.mujoco/mujoco210/bin:/usr/lib/nvidia
export REINFLOW_DIR=/home/tw559/steering_pi/KL_ReinFlow
export REINFLOW_DATA_DIR=/home/tw559/steering_pi/KL_ReinFlow/hf_cache/data-offline
export REINFLOW_LOG_DIR=/home/tw559/steering_pi/KL_ReinFlow/log
export REINFLOW_WANDB_ENTITY=terrywang330-cornell-university
export D4RL_SUPPRESS_IMPORT_ERROR=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==========="

CFG="--config-name=ft_sac_residual_flow_mlp --config-path=../cfg/gym/finetune/hopper-v2"
LOGDIR="${REINFLOW_LOG_DIR}/gym/finetune"

RUN5B_DIR="${LOGDIR}/perstep_run5b_exact_highkl"
RUN7_DIR="${LOGDIR}/perstep_run7_exact_ema"
RUN8_DIR="${LOGDIR}/perstep_run8_hutchinson_ema"

resume_arg() {
    local ckpt="$1/checkpoint/latest.pt"
    if [ -f "$ckpt" ]; then echo "resume_path=$ckpt"; else echo "resume_path=null"; fi
}

# Run 5b: Per-step exact, kl_weight=0.5 (stronger KL constraint)
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_exact \
  train.kl_weight=0.5 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 \
  train.ema_absorb_freq=0 train.ema_absorb_tau=0.01 \
  logdir="$RUN5B_DIR" $(resume_arg "$RUN5B_DIR") \
  wandb.run="perstep_run5b_exact_highkl" \
  > log/perstep_run5b_exact_highkl.log 2>&1 &
PID_5B=$!

# Run 7: Per-step exact + EMA (kl_weight=0.05, absorb every 100 actor updates)
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_exact \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 \
  train.ema_absorb_freq=100 train.ema_absorb_tau=0.01 \
  logdir="$RUN7_DIR" $(resume_arg "$RUN7_DIR") \
  wandb.run="perstep_run7_exact_ema" \
  > log/perstep_run7_exact_ema.log 2>&1 &
PID_7=$!

# Run 8: Per-step hutchinson + EMA (same KL/EMA settings as Run 7)
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_hutchinson \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 train.hutchinson_samples=4 \
  train.ema_absorb_freq=100 train.ema_absorb_tau=0.01 \
  logdir="$RUN8_DIR" $(resume_arg "$RUN8_DIR") \
  wandb.run="perstep_run8_hutchinson_ema" \
  > log/perstep_run8_hutchinson_ema.log 2>&1 &
PID_8=$!

echo "Launched: Run5b=$PID_5B  Run7=$PID_7  Run8=$PID_8"
echo "Logs: log/perstep_run{5b_exact_highkl,7_exact_ema,8_hutchinson_ema}.log"
echo "Waiting..."

wait $PID_5B $PID_7 $PID_8
echo "All runs completed."
