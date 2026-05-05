#!/bin/bash
# Round 1: Per-step KL experiments (resumes from checkpoint if available).
# Run A2: baseline (no KL) | Run 5: perstep_exact | Run 6: perstep_hutchinson
# Submit: sbatch scripts/perstep_kl_round1.sbatch

#SBATCH --requeue
#SBATCH --job-name=perstep_kl_r1
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

RUN5_DIR="${LOGDIR}/perstep_run5_exact"
RUN6_DIR="${LOGDIR}/perstep_run6_hutchinson"
RUNA2_DIR="${LOGDIR}/perstep_runA2_baseline"

resume_arg() {
    local ckpt="$1/checkpoint/latest.pt"
    if [ -f "$ckpt" ]; then echo "resume_path=$ckpt"; else echo "resume_path=null"; fi
}

handle_preempt() {
    echo "Received preemption signal, killing children and requeuing..."
    kill $PID_A $PID_5 $PID_6 2>/dev/null
    wait
    scontrol requeue ${SLURM_JOB_ID}
    exit 0
}
trap handle_preempt USR1

# Run A2: Baseline control (no KL)
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=none \
  train.kl_weight=0.0 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  logdir="$RUNA2_DIR" $(resume_arg "$RUNA2_DIR") \
  wandb.run="perstep_runA2_baseline" \
  >> log/perstep_runA2_baseline.log 2>&1 &
PID_A=$!

# Run 5: Per-step KL + exact slogdet
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_exact \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 \
  logdir="$RUN5_DIR" $(resume_arg "$RUN5_DIR") \
  wandb.run="perstep_run5_exact" \
  >> log/perstep_run5_exact.log 2>&1 &
PID_5=$!

# Run 6: Per-step KL + Hutchinson
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_hutchinson \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 train.hutchinson_samples=4 \
  logdir="$RUN6_DIR" $(resume_arg "$RUN6_DIR") \
  wandb.run="perstep_run6_hutchinson" \
  >> log/perstep_run6_hutchinson.log 2>&1 &
PID_6=$!

echo "Launched: A2=$PID_A  Run5=$PID_5  Run6=$PID_6"
echo "Logs: log/perstep_run{A2_baseline,5_exact,6_hutchinson}.log"
echo "Waiting..."

wait $PID_A $PID_5 $PID_6
echo "All round 1 runs completed."
