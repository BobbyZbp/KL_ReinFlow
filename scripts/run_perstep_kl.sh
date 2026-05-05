#!/bin/bash
# Per-step KL experiments: Run 5 (exact) and Run 6 (Hutchinson) + baseline control.
# Launch interactively on an allocated GPU node:
#   srun --jobid=<JOBID> --overlap bash scripts/run_perstep_kl.sh

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

# Fixed log dirs
RUN5_DIR="${LOGDIR}/perstep_run5_exact"
RUN6_DIR="${LOGDIR}/perstep_run6_hutchinson"
RUNA2_DIR="${LOGDIR}/perstep_runA2_baseline"

resume_arg() {
    local ckpt="$1/checkpoint/latest.pt"
    if [ -f "$ckpt" ]; then echo "resume_path=$ckpt"; else echo "resume_path=null"; fi
}

# Run A2: Baseline control (no KL)
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=none \
  train.kl_weight=0.0 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  logdir="$RUNA2_DIR" $(resume_arg "$RUNA2_DIR") \
  wandb.run="perstep_runA2_baseline" \
  > log/perstep_runA2_baseline.log 2>&1 &
PID_A=$!

# Run 5: Per-step KL + exact slogdet + fixed base
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_exact \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 \
  logdir="$RUN5_DIR" $(resume_arg "$RUN5_DIR") \
  wandb.run="perstep_run5_exact" \
  > log/perstep_run5_exact.log 2>&1 &
PID_5=$!

# Run 6: Per-step KL + Hutchinson + fixed base
python script/run.py $CFG \
  seed=42 train.alpha=0.1 train.kl_mode=perstep_hutchinson \
  train.kl_weight=0.05 train.jac_weight=0.0 \
  train.sigma_entropy_weight=0.1 train.critic_warmup_iters=1000 \
  train.perstep_fp_iters=10 train.hutchinson_samples=4 \
  logdir="$RUN6_DIR" $(resume_arg "$RUN6_DIR") \
  wandb.run="perstep_run6_hutchinson" \
  > log/perstep_run6_hutchinson.log 2>&1 &
PID_6=$!

echo "Launched: A2=$PID_A  Run5=$PID_5  Run6=$PID_6"
echo "Logs: log/perstep_run{A2_baseline,5_exact,6_hutchinson}.log"
echo "Waiting..."

wait $PID_A $PID_5 $PID_6
echo "All runs completed."
