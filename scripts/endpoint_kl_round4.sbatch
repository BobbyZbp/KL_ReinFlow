#!/bin/bash
# Round 4: Endpoint KL experiments (combos 1-4 from the 8-combination grid).
# Run 11: endpoint exact,      fixed base  (combo 1 — dangerous, may NaN)
# Run 12: endpoint hutchinson, fixed base  (combo 2 — safe logdet, inversion may diverge)
# Run 13: endpoint exact,      EMA ref     (combo 3 — EMA stabilizes inversion)
# Run 14: endpoint hutchinson, EMA ref     (combo 4 — safest endpoint variant)
# Submit: sbatch scripts/endpoint_kl_round4.sbatch

#SBATCH --requeue
#SBATCH --job-name=endpoint_kl_r4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --exclude=dgx2-compute-01,dgx2-compute-02,dgx2-compute-03,dgx2-compute-04
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --signal=B:USR1@120

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

echo "=== slurm job metadata ==="
echo "JOB_ID   = ${SLURM_JOB_ID}"
echo "JOB_NAME = ${SLURM_JOB_NAME}"
echo "NODE     = $(hostname)"
echo "GPU      = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "=========================="

CFG="--config-name=ft_sac_residual_flow_mlp --config-path=../cfg/gym/finetune/hopper-v2"
LOGDIR="${REINFLOW_LOG_DIR}/gym/finetune"

RUN11_DIR="${LOGDIR}/endpoint_run11_exact_fixed"
RUN12_DIR="${LOGDIR}/endpoint_run12_hutchinson_fixed"
RUN13_DIR="${LOGDIR}/endpoint_run13_exact_ref"
RUN14_DIR="${LOGDIR}/endpoint_run14_hutchinson_ref"

resume_arg() {
    local ckpt="$1/checkpoint/latest.pt"
    if [ -f "$ckpt" ]; then echo "resume_path=$ckpt"; else echo "resume_path=null"; fi
}

handle_preempt() {
    echo "Received preemption signal, killing children and requeuing..."
    kill $PID_11 $PID_12 $PID_13 $PID_14 2>/dev/null
    wait
    scontrol requeue ${SLURM_JOB_ID}
    exit 0
}
trap handle_preempt USR1

# Run 11: Endpoint exact, fixed base (combo 1)
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=exact \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.ema_absorb_freq=0 \
  logdir="$RUN11_DIR" $(resume_arg "$RUN11_DIR") \
  wandb.run="endpoint_run11_exact_fixed" \
  >> log/endpoint_run11_exact_fixed.log 2>&1 &
PID_11=$!

# Run 12: Endpoint hutchinson, fixed base (combo 2)
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=hutchinson \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.hutchinson_samples=4 \
  train.ema_absorb_freq=0 \
  logdir="$RUN12_DIR" $(resume_arg "$RUN12_DIR") \
  wandb.run="endpoint_run12_hutchinson_fixed" \
  >> log/endpoint_run12_hutchinson_fixed.log 2>&1 &
PID_12=$!

# Run 13: Endpoint exact, EMA reference (combo 3)
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=exact_ref \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.ema_absorb_freq=100 train.ema_absorb_tau=0.01 \
  logdir="$RUN13_DIR" $(resume_arg "$RUN13_DIR") \
  wandb.run="endpoint_run13_exact_ref" \
  >> log/endpoint_run13_exact_ref.log 2>&1 &
PID_13=$!

# Run 14: Endpoint hutchinson, EMA reference (combo 4)
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=hutchinson_ref \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.hutchinson_samples=4 \
  train.ema_absorb_freq=100 train.ema_absorb_tau=0.01 \
  logdir="$RUN14_DIR" $(resume_arg "$RUN14_DIR") \
  wandb.run="endpoint_run14_hutchinson_ref" \
  >> log/endpoint_run14_hutchinson_ref.log 2>&1 &
PID_14=$!

echo "Launched: Run11=$PID_11  Run12=$PID_12  Run13=$PID_13  Run14=$PID_14"
echo "Logs: log/endpoint_run{11_exact_fixed,12_hutchinson_fixed,13_exact_ref,14_hutchinson_ref}.log"
echo "Waiting..."

wait $PID_11 $PID_12 $PID_13 $PID_14
echo "All round 4 runs completed."
