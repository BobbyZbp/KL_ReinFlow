# CLAUDE.md — KL_ReinFlow Project

## Environment Rules

- **NEVER run compute on the login node.** All Python scripts (training, diagnostics, anything importing torch) must run on a GPU compute node via `salloc` or `sbatch`.
- To get an interactive GPU session: `salloc -p gpu-interactive --gres=gpu:1 --cpus-per-task=4 --mem=64G -t 4:00:00`
- Conda environment: `reinflow`
- Activate before running: `conda activate reinflow`

## Project Structure

- Flow-matching RL fine-tuning with SAC + residual velocity + KL regularization
- Base policy: frozen pretrained ReFlow (v_base, 553K params)
- Trainable: residual velocity (v_res, 24K params) + critic + sigma head
- Environment: Hopper-v2 (D4RL), action dim 3, obs dim 11

## Key Paths

- Checkpoints: `log/gym/finetune/{run_name}/checkpoint/`
- Full checkpoints include replay buffer (300-500MB); use `model_only/` subdirectory for lightweight model-only files
- Base policy: `log/gym/pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_D4RL_42/state_40.pt`
- Normalization: `hf_cache/data-offline/gym/hopper-medium-v2/normalization.npz`

## Run Names

- `perstep_runA2_baseline` — SAC without KL (baseline)
- `ablation_runB_reward_penalty` — KL as reward penalty
- `perstep_run5b_exact_highkl` — per-step exact KL
- `perstep_run7_exact_ema` — per-step exact + EMA reference
- `perstep_run6_hutchinson` — per-step Hutchinson
