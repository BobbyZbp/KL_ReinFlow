#!/bin/bash
# Round 5: Per-step KL with critic grad clip fix (critic_grad_clip=null).
# 2x2 grid: {exact, hutchinson} x {fixed base, EMA ref}
# Run 15: perstep_exact,         fixed base
# Run 16: perstep_hutchinson,    fixed base
# Run 17: perstep_exact_ref,     EMA reference
# Run 18: perstep_hutchinson_ref, EMA reference
# Submit: sbatch scripts/perstep_kl_round5_clipfix.sbatch

#SBATCH --requeue
#SBATCH --job-name=perstep_kl_r5
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

RUN15_DIR="${LOGDIR}/perstep_run15_exact_clipfix"
RUN16_DIR="${LOGDIR}/perstep_run16_hutchinson_clipfix"
RUN17_DIR="${LOGDIR}/perstep_run17_exact_ref_clipfix"
RUN18_DIR="${LOGDIR}/perstep_run18_hutchinson_ref_clipfix"

resume_arg() {
    local ckpt="$1/checkpoint/latest.pt"
    if [ -f "$ckpt" ]; then echo "resume_path=$ckpt"; else echo "resume_path=null"; fi
}

handle_preempt() {
    echo "Received preemption signal, killing children and requeuing..."
    kill $PID_15 $PID_16 $PID_17 $PID_18 2>/dev/null
    wait
    scontrol requeue ${SLURM_JOB_ID}
    exit 0
}
trap handle_preempt USR1

# Run 15: Per-step exact, fixed base, no critic clip
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_exact \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 \
  train.critic_grad_clip=null train.actor_grad_clip=1.0 \
  train.ema_absorb_freq=0 \
  logdir="$RUN15_DIR" $(resume_arg "$RUN15_DIR") \
  wandb.run="perstep_run15_exact_clipfix" \
  >> log/perstep_run15_exact_clipfix.log 2>&1 &
PID_15=$!

# Run 16: Per-step hutchinson, fixed base, no critic clip
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_hutchinson \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 train.hutchinson_samples=4 \
  train.critic_grad_clip=null train.actor_grad_clip=1.0 \
  train.ema_absorb_freq=0 \
  logdir="$RUN16_DIR" $(resume_arg "$RUN16_DIR") \
  wandb.run="perstep_run16_hutchinson_clipfix" \
  >> log/perstep_run16_hutchinson_clipfix.log 2>&1 &
PID_16=$!

# Run 17: Per-step exact, EMA reference, no critic clip
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_exact_ref \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 \
  train.critic_grad_clip=null train.actor_grad_clip=1.0 \
  train.ema_absorb_freq=100 train.ema_absorb_tau=0.01 \
  logdir="$RUN17_DIR" $(resume_arg "$RUN17_DIR") \
  wandb.run="perstep_run17_exact_ref_clipfix" \
  >> log/perstep_run17_exact_ref_clipfix.log 2>&1 &
PID_17=$!

# Run 18: Per-step hutchinson, EMA reference, no critic clip
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_hutchinson_ref \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 train.hutchinson_samples=4 \
  train.critic_grad_clip=null train.actor_grad_clip=1.0 \
  train.ema_absorb_freq=100 train.ema_absorb_tau=0.01 \
  logdir="$RUN18_DIR" $(resume_arg "$RUN18_DIR") \
  wandb.run="perstep_run18_hutchinson_ref_clipfix" \
  >> log/perstep_run18_hutchinson_ref_clipfix.log 2>&1 &
PID_18=$!

echo "Launched: Run15=$PID_15  Run16=$PID_16  Run17=$PID_17  Run18=$PID_18"
echo "Logs: log/perstep_run{15_exact_clipfix,16_hutchinson_clipfix,17_exact_ref_clipfix,18_hutchinson_ref_clipfix}.log"
echo "Waiting..."

wait $PID_15 $PID_16 $PID_17 $PID_18
echo "All round 5 runs completed."
