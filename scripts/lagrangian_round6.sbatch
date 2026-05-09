#!/bin/bash
# Round 6: Auto-entropy tuning + Lagrangian dual KL
# 3 runs in parallel on a single GPU:
#   Run A: Baseline — no KL (auto-entropy only)
#   Run B: Per-step exact KL, fixed base (Lagrangian dual, no EMA)
#   Run C: Per-step exact KL, EMA reference (Lagrangian dual, EMA ref)
# Submit: sbatch scripts/lagrangian_round6.sbatch

#SBATCH --requeue
#SBATCH --job-name=lagr_r6
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

# Common flags: auto-entropy tuning, no sigma_ent_loss, critic warmup
COMMON="seed=42 \
  train.auto_entropy_tuning=true \
  train.init_alpha=0.1 \
  train.target_entropy=-12 \
  train.alpha_lr=3e-4 \
  train.sigma_entropy_weight=0.0 \
  train.critic_warmup_iters=5000 \
  train.critic_grad_clip=null \
  train.actor_grad_clip=1.0 \
  train.jac_weight=0.0 \
  train.perstep_fp_iters=10"

RUNA_DIR="${LOGDIR}/lagr_runA_baseline"
RUNB_DIR="${LOGDIR}/lagr_runB_exact"
RUNC_DIR="${LOGDIR}/lagr_runC_exact_ref"

resume_arg() {
    local ckpt="$1/checkpoint/latest.pt"
    if [ -f "$ckpt" ]; then echo "resume_path=$ckpt"; else echo "resume_path=null"; fi
}

handle_preempt() {
    echo "Received preemption signal, killing children and requeuing..."
    kill $PID_A $PID_B $PID_C 2>/dev/null
    wait
    scontrol requeue ${SLURM_JOB_ID}
    exit 0
}
trap handle_preempt USR1

# Run A: Baseline (no KL)
python script/run.py $CFG $COMMON \
  train.kl_mode=none \
  train.kl_weight=0.0 \
  train.kl_lagrangian=false \
  train.ema_absorb_freq=0 \
  logdir="$RUNA_DIR" $(resume_arg "$RUNA_DIR") \
  wandb.run="lagr_runA_baseline" \
  >> log/lagr_runA_baseline.log 2>&1 &
PID_A=$!

# Run B: Per-step exact KL, fixed base, Lagrangian dual
python script/run.py $CFG $COMMON \
  train.kl_mode=perstep_exact \
  train.kl_lagrangian=true \
  train.kl_target_epsilon=1.0 \
  train.kl_init_eta=1.0 \
  train.kl_eta_lr=1e-3 \
  train.ema_absorb_freq=0 \
  logdir="$RUNB_DIR" $(resume_arg "$RUNB_DIR") \
  wandb.run="lagr_runB_exact" \
  >> log/lagr_runB_exact.log 2>&1 &
PID_B=$!

# Run C: Per-step exact KL, EMA reference, Lagrangian dual
python script/run.py $CFG $COMMON \
  train.kl_mode=perstep_exact_ref \
  train.kl_lagrangian=true \
  train.kl_target_epsilon=1.0 \
  train.kl_init_eta=1.0 \
  train.kl_eta_lr=1e-3 \
  train.ema_absorb_freq=100 train.ema_absorb_tau=0.01 \
  logdir="$RUNC_DIR" $(resume_arg "$RUNC_DIR") \
  wandb.run="lagr_runC_exact_ref" \
  >> log/lagr_runC_exact_ref.log 2>&1 &
PID_C=$!

echo "Launched: RunA=$PID_A  RunB=$PID_B  RunC=$PID_C"
echo "Logs: log/lagr_run{A_baseline,B_exact,C_exact_ref}.log"
echo "Waiting..."

wait $PID_A $PID_B $PID_C
echo "All round 6 runs completed."
