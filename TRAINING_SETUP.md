# SAC + Residual Flow + Exact KL: Training Setup

## Task

**Environment**: Hopper-v2 (MuJoCo locomotion via D4RL)
- **Observation**: 11-dim state (qpos/qvel of hopper joints), normalized to roughly [-3.5, 3.6] per dim via `normalization.npz`
- **Action**: 3-dim continuous, range [-1, 1]
- **Horizon**: 4 action steps per inference (multi-step action chunking via `MultiStep` wrapper)
- **Reward**: Dense MuJoCo locomotion reward (forward velocity + alive bonus - control cost), summed over `act_steps=4` sub-steps per environment step. No additional reward scaling (`scale_reward_factor=1.0`).
- **Success threshold**: best per-step reward >= 3.0
- **Episode length**: 1000 steps max

**Embodiment**: 2D hopper (3 joints: thigh, leg, foot)

## Algorithm

**SAC + Flow Matching + Residual Velocity + Exact Deterministic KL**

### Policy architecture

The policy is a flow-matching model that generates actions via a K-step forward ODE (K=4, dt=0.25):

```
a^0 ~ N(0, I)
for k = 0..K-2:     (3 deterministic steps)
    a^{k+1} = a^k + [v_base(a^k, t_k, s) + v_res(a^k, t_k, s)] * dt
a^K = a^{K-1} + [v_base + v_res](a^{K-1}, t_{K-1}, s) * dt + sigma_phi(s) * eps
```

- **v_base** (frozen): Pre-trained ReFlow velocity field on D4RL Hopper-medium-v2 offline data (40 epochs, `state_40.pt`, EMA weights). Never updated during fine-tuning.
- **v_res** (trainable): Residual velocity correction. Zero-initialized (last linear layer) so the combined policy starts identical to the base.
- **sigma_phi** (trainable): Observation-conditioned exploration noise, bounded to [0.05, 0.15] via sigmoid.

### Loss functions

**Critic loss** (standard SAC, every iteration):
```
L_critic = MSE(Q(s,a), r + gamma * (1-d) * min Q_target(s', a'))
```
No entropy term in the target (entropy bonus is disabled; KL serves as the regularizer).

**Actor loss** (delayed, every 2 iterations):
```
L_actor = -min(Q1, Q2)(s, a^K)  +  kl_weight * KL  +  jac_weight * ||J_res||_F^2
```

- **SAC term**: Maximize Q-value of reparameterized actions (gradients flow through the full K-step ODE + final Gaussian).
- **KL term**: Exact deterministic KL divergence at the penultimate point a^{K-1}, computed via change of variables on the ODE pushforward density:
  - Forward: log p_theta(a^{K-1}) = log p_0(a^0) - sum_k log|det(I + J_combined * dt)|
  - Backward: log p_base(a^{K-1}) via fixed-point inversion (30 iterations) to recover the base trajectory, then change of variables along the base ODE.
  - KL = log p_theta - log p_base, clamped to [-50, 50] nats.
- **Jacobian regularizer**: Frobenius norm of the residual velocity Jacobian, averaged over ODE steps. Keeps the residual flow well-conditioned (replaces the need for intermediate action clamping, which would break change of variables).

### Key design choices

1. **No entropy bonus**: The standard SAC entropy term alpha * H[pi] is removed. The marginal log-prob of the full flow policy is intractable without per-sample Jacobian computation (which is expensive). The KL regularizer serves the same role: it penalizes deviation from the well-behaved base policy, preventing mode collapse.

2. **Reparameterized gradients**: Actions are produced via the ODE + Gaussian reparameterization trick. No importance sampling or REINFORCE estimators needed - gradients flow directly through the sampling process.

3. **Fixed-point backward inversion**: To evaluate the base policy's density at a point produced by the combined policy, we invert the base ODE using fixed-point iteration (a^k = a^{k+1} - dt * v_base(a^k, t_k)). 30 iterations achieves <1% round-trip error on in-distribution observations.

4. **Jacobian evaluated at inverted point**: Critical correctness detail - the Jacobian for change-of-variables must be evaluated at the point *after* fixed-point convergence (a^k), not at the pre-inversion point (a^{k+1}).

## Models

| Component | Architecture | Params | Trainable | Initialization |
|-----------|-------------|--------|-----------|----------------|
| v_base | ResidualMLP [512, 512, 512], ReLU | 0.553M | No (frozen) | Pre-trained ReFlow checkpoint (EMA weights, 40 epochs on D4RL offline data) |
| v_res | MLP [128, 128], ReLU | 0.024M | Yes | Zero-init last layer (policy starts identical to base) |
| Q1, Q2 | MLP [256, 256, 256], Mish, LayerNorm | 0.279M total | Yes | Random (PyTorch default Kaiming); learns from scratch via TD |
| Q1_target, Q2_target | Copy of Q1, Q2 | same | No (EMA, tau=0.005) | Deep copy of Q1, Q2 at init |
| sigma_head | MLP [64, 64], SiLU, sigmoid output | 0.006M | Yes | Zero-init last layer (output starts at mid-range ~0.1) |

### Base policy checkpoint

`state_40.pt` from `ReinFlow/ReinFlow-data-checkpoints-logs` on HuggingFace. This is the **final** checkpoint (epoch 40/40) of ReFlow pre-training on the D4RL `hopper-medium-v2` offline dataset:
- Architecture: FlowMLP [512, 512, 512] with residual connections, ReLU
- Training: behavior cloning via flow matching, lr=1e-3 cosine schedule, batch_size=128
- EMA decay=0.995; the `ema` weights are loaded into `v_base`

### Inputs

- v_base, v_res: `(a, t, cond)` where a is (B, 4, 3), t is (B,), cond is {"state": (B, 1, 11)}
- Q(s,a): concatenation of flattened state (11) and action (12) -> scalar
- sigma_head: flattened state (11) -> (B, 4, 3) bounded noise scale

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| gamma | 0.99 | Discount factor |
| actor_lr | 3e-5 | Small to avoid destabilizing the ODE |
| critic_lr | 3e-4 | Standard |
| target_ema_rate | 0.005 | Polyak averaging for target critics |
| batch_size | 256 | From replay buffer |
| buffer_size | 1M | FIFO replay |
| n_explore_steps | 1000 | Random actions before training (reduced: base policy is already competent) |
| critic_update_freq | 1 | Every iteration |
| actor_update_freq | 2 | Delayed (every 2 iterations) |
| kl_weight | 0.05 | On exact deterministic KL |
| jac_weight | 0.01 | On residual Jacobian Frobenius norm |
| alpha | 0.0 | Entropy bonus disabled |
| K (denoising_steps) | 4 | ODE integration steps |
| sigma_min, sigma_max | 0.05, 0.15 | Bounds for sigma_head output |
| backward_fp_iters | 30 | Fixed-point iterations for base ODE inversion |
| n_envs | 40 | Parallel environments (config default; pilot used 1 due to async deadlock) |
| n_train_itr | 200K | Total training iterations |
| grad clip (critic) | 10.0 | Defends against Q-target divergence |
| grad clip (actor) | 1.0 | Chains through K-1 ODE Jacobians |

### Timing (B=256, K=4, D=12, RTX A5500)

| Operation | Time |
|-----------|------|
| forward_with_logdet (3 Jacobians via vmap(jacrev)) | ~211 ms |
| Actor step total (forward + backward base + Jac reg) | ~110-150 ms per iter |
| Critic-only step | ~20-30 ms per iter |

## Data

- **Pre-training**: D4RL Hopper-medium-v2 offline dataset (used to train v_base, 40 epochs)
- **Fine-tuning**: Online interaction with the Hopper-v2 environment, collected into a FIFO replay buffer
- **Normalization**: `normalization.npz` contains per-dim obs min/max (used by `mujoco_locomotion_lowdim` wrapper to normalize observations to roughly [-1, 1])

## Changes made (from friend's original implementation)

The original `loss_actor` used a **standard SAC entropy bonus** (`alpha * log pi`) approximated via the final Gaussian step. We replaced it with the **exact deterministic KL** that was already implemented in `compute_kl_and_action()` but not wired in.

### Files changed

1. **`model/flow/ft_sac/sac_residual_flow.py`** — replaced `loss_actor`:
   - Removed: entropy bonus via `eps.pow(2)` and `sigma.log()` approximation
   - Added: calls `compute_kl_and_action()` to get exact KL and Jacobian regularizer
   - Actor loss is now: `-Q_min + kl_weight * KL + jac_weight * ||J_res||_F^2`
   - Cleaned up unused imports (`math`, `Tuple`, `Any`)

2. **`cfg/gym/finetune/hopper-v2/ft_sac_residual_flow_mlp.yaml`**:
   - Set `alpha: 0.0` (entropy bonus disabled)
   - Uncommented `kl_weight: 0.05` and `jac_weight: 0.01`
   - Reduced `n_explore_steps: 5000 -> 1000` (base policy already competent)

3. **`agent/finetune/reinflow/train_sac_residual_flow_agent.py`**:
   - Removed stale "KL integration pending" comments
   - Updated log line to print `kl_w` and `jac_w` values

### Pilot run results (2000 iters, 1 env, RTX A5500)

| Metric | Value | Status |
|--------|-------|--------|
| actor/kl_mean | 0.45 nats | Healthy (was 1e17 in friend's previous run) |
| actor/loss_kl | 0.023 | Bounded |
| actor/loss_jac | 0.0 | v_res still near zero early in training |
| actor/loss_sac | -5.19 | Q-values growing |
| actor/q_mean | 5.19 | Rising |
| actor/sigma_mean | 0.100 | Mid-range of [0.05, 0.15] |
| avg episode reward | 42.85 | Low (very early, 1 env, BC baseline ~1500-2000) |
| loss - critic | 32.8 | Bounded |
| num episodes completed | 36 | Episodes completing normally |

## Environment setup

### 1. Create conda env and install packages

```bash
conda create -n reinflow python=3.8 -y
conda activate reinflow

# PyTorch with CUDA
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# MuJoCo 210
mkdir -p ~/.mujoco && cd ~/.mujoco
wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz && rm mujoco210-linux-x86_64.tar.gz

# GL dependencies (no sudo needed)
conda install -c conda-forge glew mesalib -y
conda install -c menpo glfw3 -y

# Install ReinFlow with gym extras
cd ~/steering_pi/KL_ReinFlow
pip install cython"<3.0.0" numpy
pip install -e ".[gym]"
```

### 2. Download pretrained checkpoint and normalization

```bash
conda activate reinflow
python3 -c "
from huggingface_hub import hf_hub_download
REPO = 'ReinFlow/ReinFlow-data-checkpoints-logs'
HF = 'hf_cache'
hf_hub_download(repo_id=REPO, repo_type='dataset',
    filename='log/log_gym_d4rl_pretrained/hopper-medium-v2/1-ReFlow/state_40.pt',
    local_dir=HF)
hf_hub_download(repo_id=REPO, repo_type='dataset',
    filename='data-offline/gym_d4rl/hopper-medium-v2/normalization.npz',
    local_dir=HF)
"
```

### 3. Run on a GPU compute node

```bash
# Get a GPU node
salloc -p gpu --gres=gpu:1 --constraint="a6000|a40|a5500" --cpus-per-task=8 --mem=128G -t 4:00:00

# On the compute node:
conda activate reinflow
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
export D4RL_SUPPRESS_IMPORT_ERROR=1
export HYDRA_FULL_ERROR=1
export REINFLOW_DIR=~/steering_pi/KL_ReinFlow
export REINFLOW_DATA_DIR=~/steering_pi/KL_ReinFlow/data
export REINFLOW_LOG_DIR=~/steering_pi/KL_ReinFlow/log
export REINFLOW_WANDB_ENTITY=none
cd ~/steering_pi/KL_ReinFlow

CKPT=hf_cache/log/log_gym_d4rl_pretrained/hopper-medium-v2/1-ReFlow/state_40.pt
NORM=hf_cache/data-offline/gym_d4rl/hopper-medium-v2/normalization.npz
```

### 4. Smoke test

```bash
python3 script/test_sac_residual_flow.py
# Expected: "ALL CHECKS PASSED", KL=0 in checks 4a/4b
```

### 5. Pilot run (short, verify system works end-to-end)

```bash
python3 script/run.py \
    --config-path=../cfg/gym/finetune/hopper-v2 \
    --config-name=ft_sac_residual_flow_mlp \
    base_policy_path=$CKPT \
    normalization_path=$NORM \
    train.n_train_itr=2000 \
    train.n_explore_steps=200 \
    env.n_envs=1 \
    wandb.offline_mode=true
```

### 6. Full run

```bash
python3 script/run.py \
    --config-path=../cfg/gym/finetune/hopper-v2 \
    --config-name=ft_sac_residual_flow_mlp \
    base_policy_path=$CKPT \
    normalization_path=$NORM \
    wandb.offline_mode=true
```

### Known issues

- **Async env deadlock**: `n_envs>1` with `asynchronous=True` can deadlock after ~450 iters. Pilot runs should use `env.n_envs=1`. For full runs, debug the async multiprocessing or set `env.n_envs=1` (slower but reliable).
- **`MUJOCO_GL=egl` required on GPU nodes**: The cluster lacks `libosmesa6-dev`, so mujoco_py must compile with EGL backend. Set `export MUJOCO_GL=egl` before the first `import mujoco_py`.
- **`import d4rl` vs `import d4rl.gym_mujoco`**: With `D4RL_SUPPRESS_IMPORT_ERROR=1`, `import d4rl` silently skips MuJoCo env registration. The ReinFlow code correctly uses `import d4rl.gym_mujoco` which works.

## File Structure

```
KL_ReinFlow/
├── agent/finetune/reinflow/
│   └── train_sac_residual_flow_agent.py    # Training loop (replay buffer, env interaction, logging)
├── model/
│   ├── flow/
│   │   ├── mlp_flow.py                     # FlowMLP (base velocity field architecture)
│   │   └── ft_sac/
│   │       └── sac_residual_flow.py        # SACResidualFlow + SigmaHead (policy, KL, losses)
│   └── common/
│       ├── critic.py                       # CriticObsAct (twin Q-networks)
│       └── mlp.py                          # MLP / ResidualMLP building blocks
├── cfg/
│   └── gym/
│       ├── finetune/hopper-v2/
│       │   └── ft_sac_residual_flow_mlp.yaml   # Fine-tuning config
│       └── pretrain/hopper-medium-v2/
│           └── pre_reflow_mlp.yaml              # Pre-training config (produced state_40.pt)
├── script/
│   ├── run.py                              # Hydra entry point
│   └── test_sac_residual_flow.py           # Smoke test (6 checks)
├── hf_cache/                               # Downloaded checkpoints + normalization
│   ├── log/.../state_40.pt                 # Pre-trained base policy
│   └── data-offline/.../normalization.npz  # Observation normalization
└── TRAINING_SETUP.md                       # This file
```

### Key methods in `sac_residual_flow.py`

- `sample_action()`: Forward ODE sampling (K-1 deterministic + 1 noisy step)
- `forward_with_logdet()`: Forward ODE with per-step Jacobian log-determinants for theta density
- `_backward_base_recover_trajectory()`: Fixed-point inversion of base ODE
- `backward_base_logprob()`: Base density via change of variables on recovered trajectory
- `compute_kl_and_action()`: Combines forward density, backward base density, and Jacobian reg
- `loss_actor()`: Calls `compute_kl_and_action`, combines -Q + KL + Jac reg
- `loss_critic()`: Standard SAC critic loss with target network
