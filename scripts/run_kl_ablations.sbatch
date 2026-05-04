#!/bin/bash
# KL stabilization ablation experiments (5 runs in parallel on one GPU).
# Submit with:  sbatch scripts/run_kl_ablations.sbatch
# Auto-resumes from latest checkpoint if preempted and requeued.

#SBATCH --requeue
#SBATCH --job-name=kl_ablations
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --signal=B:USR1@120

set -euo pipefail

cd /home/tw559/steering_pi/KL_ReinFlow
mkdir -p slurm_logs log

# ---- environment ----
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

# ---- fixed log dirs per run (persist across requeues) ----
RUNA_DIR="${LOGDIR}/ablation_runA_baseline"
RUNB_DIR="${LOGDIR}/ablation_runB_reward_penalty"
RUNC_DIR="${LOGDIR}/ablation_runC_hutchinson"
RUND_DIR="${LOGDIR}/ablation_runD_detach_logdet"
RUNE_DIR="${LOGDIR}/ablation_runE_vres_l2"

# Helper: find latest.pt for a run dir, return resume_path arg or null
resume_arg() {
    local ckpt="$1/checkpoint/latest.pt"
    if [ -f "$ckpt" ]; then
        echo "resume_path=$ckpt"
    else
        echo "resume_path=null"
    fi
}

# ---- on preemption: kill children, requeue ----
handle_preempt() {
    echo "Received preemption signal, killing children and requeuing..."
    kill $PID_A $PID_B $PID_C $PID_D $PID_E 2>/dev/null
    wait
    scontrol requeue ${SLURM_JOB_ID}
    exit 0
}
trap handle_preempt USR1

# Run A: Baseline (no KL, alpha=0.1)
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=none \
  train.kl_weight=0.0 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  logdir="$RUNA_DIR" $(resume_arg "$RUNA_DIR") \
  wandb.run="runA_baseline" \
  > log/runA_baseline.log 2>&1 &
PID_A=$!

# Run B: KL as reward penalty
python script/run.py $CFG \
  seed=43 train.alpha=0.1 train.kl_mode=reward_penalty \
  train.kl_weight=0.0 train.jac_weight=0.0 train.kl_reward_weight=0.05 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  logdir="$RUNB_DIR" $(resume_arg "$RUNB_DIR") \
  wandb.run="runB_kl_reward_penalty" \
  > log/runB_kl_reward_penalty.log 2>&1 &
PID_B=$!

# Run C: Hutchinson trace (with detached base logprob)
python script/run.py $CFG \
  seed=44 train.alpha=0.1 train.kl_mode=hutchinson \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.hutchinson_samples=1 \
  train.critic_warmup_iters=1000 \
  logdir="$RUNC_DIR" $(resume_arg "$RUNC_DIR") \
  wandb.run="runC_hutchinson" \
  > log/runC_hutchinson.log 2>&1 &
PID_C=$!

# Run D: Detach logdet (both theta and base detached — control)
python script/run.py $CFG \
  seed=45 train.alpha=0.1 train.kl_mode=detach_logdet \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  logdir="$RUND_DIR" $(resume_arg "$RUND_DIR") \
  wandb.run="runD_detach_logdet" \
  > log/runD_detach_logdet.log 2>&1 &
PID_D=$!

# Run E: v_res L2 penalty
python script/run.py $CFG \
  seed=46 train.alpha=0.1 train.kl_mode=vres_l2 \
  train.kl_weight=0.0 train.jac_weight=0.0 train.vres_l2_weight=0.01 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  logdir="$RUNE_DIR" $(resume_arg "$RUNE_DIR") \
  wandb.run="runE_vres_l2" \
  > log/runE_vres_l2.log 2>&1 &
PID_E=$!

echo "Launched: A=$PID_A B=$PID_B C=$PID_C D=$PID_D E=$PID_E"
echo "Waiting for all to finish..."

wait $PID_A $PID_B $PID_C $PID_D $PID_E
echo "All runs completed."
