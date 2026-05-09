#!/bin/bash
# Re-run per-step det diagnostic with fixed obs collection (on-policy per checkpoint).
# Submit: sbatch scripts/diagnose_perstep_v2.sbatch

#SBATCH --job-name=diag_det_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --exclude=dgx2-compute-01,dgx2-compute-02,dgx2-compute-03,dgx2-compute-04
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

cd /home/tw559/steering_pi/KL_ReinFlow
mkdir -p slurm_logs log

export PATH=/home/tw559/.conda/envs/reinflow/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/home/tw559/.mujoco/mujoco210/bin:/usr/lib/nvidia
export REINFLOW_DATA_DIR=/home/tw559/steering_pi/KL_ReinFlow/hf_cache/data-offline
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl

echo "=== slurm job metadata ==="
echo "JOB_ID   = ${SLURM_JOB_ID}"
echo "NODE     = $(hostname)"
echo "GPU      = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "=========================="

echo ""
echo "=========================================="
echo "  Run A2 (baseline, no KL)"
echo "=========================================="
python script/diagnose_perstep_det.py \
  --run A2 \
  --iters 0,5000,10000,20000,40000,60000,80000 \
  --n_samples 256 \
  --fp_iters 10 \
  --device cuda:0 \
  2>&1 | tee log/diagnose_perstep_det_v2_A2.log

echo ""
echo "=========================================="
echo "  Run B (KL reward penalty)"
echo "=========================================="
python script/diagnose_perstep_det.py \
  --run B \
  --iters 0,5000,10000,20000,40000,60000,80000,100000,150000,199999 \
  --n_samples 256 \
  --fp_iters 10 \
  --device cuda:0 \
  2>&1 | tee log/diagnose_perstep_det_v2_B.log

echo ""
echo "All diagnostics complete."
