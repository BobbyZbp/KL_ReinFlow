#!/bin/bash
# Round 3: EMA reference policy experiments (LoRA-style ref_v_res).
# Run 9:  perstep_exact_ref      kl=0.05, ema_freq=100, ema_tau=0.01
# Run 10: perstep_hutchinson_ref kl=0.05, ema_freq=100, ema_tau=0.01
# Submit: sbatch scripts/perstep_kl_round3_ref.sbatch

#SBATCH --requeue
#SBATCH --job-name=perstep_kl_r3_ref
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

RUN9_DIR="${LOGDIR}/perstep_run9_exact_ref"
RUN10_DIR="${LOGDIR}/perstep_run10_hutchinson_ref"

resume_arg() {
    local ckpt="$1/checkpoint/latest.pt"
    if [ -f "$ckpt" ]; then echo "resume_path=$ckpt"; else echo "resume_path=null"; fi
}

handle_preempt() {
    echo "Received preemption signal, killing children and requeuing..."
    kill $PID_9 $PID_10 2>/dev/null
    wait
    scontrol requeue ${SLURM_JOB_ID}
    exit 0
}
trap handle_preempt USR1

# Run 9: Per-step exact + EMA reference policy
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_exact_ref \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 \
  train.ema_absorb_freq=100 train.ema_absorb_tau=0.01 \
  logdir="$RUN9_DIR" $(resume_arg "$RUN9_DIR") \
  wandb.run="perstep_run9_exact_ref" \
  >> log/perstep_run9_exact_ref.log 2>&1 &
PID_9=$!

# Run 10: Per-step Hutchinson + EMA reference policy
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_hutchinson_ref \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 train.hutchinson_samples=4 \
  train.ema_absorb_freq=100 train.ema_absorb_tau=0.01 \
  logdir="$RUN10_DIR" $(resume_arg "$RUN10_DIR") \
  wandb.run="perstep_run10_hutchinson_ref" \
  >> log/perstep_run10_hutchinson_ref.log 2>&1 &
PID_10=$!

echo "Launched: Run9=$PID_9  Run10=$PID_10"
echo "Logs: log/perstep_run{9_exact_ref,10_hutchinson_ref}.log"
echo "Waiting..."

wait $PID_9 $PID_10
echo "All round 3 runs completed."
