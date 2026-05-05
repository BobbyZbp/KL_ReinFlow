#!/bin/bash
# Round 2b: KL weight sweep (runs 7 and 8 only — Run 5b already running).
# Run 7: perstep_exact kl=0.2 | Run 8: perstep_exact kl=1.0
# Submit: sbatch scripts/perstep_kl_round2b.sbatch

#SBATCH --requeue
#SBATCH --job-name=perstep_kl_r2b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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

RUN7_DIR="${LOGDIR}/perstep_run7_exact_kl02"
RUN8_DIR="${LOGDIR}/perstep_run8_exact_kl10"

resume_arg() {
    local ckpt="$1/checkpoint/latest.pt"
    if [ -f "$ckpt" ]; then echo "resume_path=$ckpt"; else echo "resume_path=null"; fi
}

handle_preempt() {
    echo "Received preemption signal, killing children and requeuing..."
    kill $PID_7 $PID_8 2>/dev/null
    wait
    scontrol requeue ${SLURM_JOB_ID}
    exit 0
}
trap handle_preempt USR1

# Run 7: Per-step exact, kl_weight=0.2
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_exact \
  train.kl_weight=0.2 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 \
  logdir="$RUN7_DIR" $(resume_arg "$RUN7_DIR") \
  wandb.run="perstep_run7_exact_kl02" \
  > log/perstep_run7_exact_kl02.log 2>&1 &
PID_7=$!

# Run 8: Per-step exact, kl_weight=1.0
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_exact \
  train.kl_weight=1.0 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 \
  logdir="$RUN8_DIR" $(resume_arg "$RUN8_DIR") \
  wandb.run="perstep_run8_exact_kl10" \
  > log/perstep_run8_exact_kl10.log 2>&1 &
PID_8=$!

echo "Launched: Run7=$PID_7  Run8=$PID_8"
echo "Logs: log/perstep_run{7_exact_kl02,8_exact_kl10}.log"
echo "Waiting..."

wait $PID_7 $PID_8
echo "All round 2b runs completed."
