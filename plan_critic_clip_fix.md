# Plan: Fix Critic Gradient Clipping

## Problem

The critic gradient clip `max_norm=10` is non-standard for SAC and excessively aggressive:
- Median critic grad_norm is ~10,000, so the effective learning rate is `3e-4 * (10/10000) = 3e-7`
- Standard SAC (CleanRL, SB3, SpinningUp) uses **no gradient clipping** on the critic
- This creates a critic lag → policy oscillation feedback loop: the actor moves to high-reward regions, but the critic can't evaluate them fast enough, causing reward oscillations

## Root Cause

1. **Deeper architecture** than standard SAC: [256, 256, 256] with LayerNorm + Mish (vs standard [256, 256] with ReLU, no LayerNorm)
2. **Higher input dim** (23 vs 14) due to multi-step action chunking (4 steps × 3 action dims)
3. **No weight decay** — weights grow freely, increasing the network Jacobian norm
4. **max_norm=10 is a holdover** from early debugging when the critic was NaN-ing (which was actually caused by missing SAC entropy, now fixed with alpha=0.1)

## Changes

### 1. Make grad clip configurable via YAML (`train_sac_residual_flow_agent.py`)

Add config fields `critic_grad_clip` and `actor_grad_clip` with sensible defaults:
- **Critic**: default `null` (no clipping, matching standard SAC)
- **Actor**: keep `1.0` (the actor backprops through the ODE which genuinely amplifies gradients from slogdet backward)

In the training loop:
- If `critic_grad_clip` is null/0, skip `clip_grad_norm_` entirely (just log the norm)
- If set, clip as before
- Same pattern for actor

### 2. Update YAML config (`ft_sac_residual_flow_mlp.yaml`)

Add under `train:`:
```yaml
critic_grad_clip: null    # no clipping (standard SAC); set to e.g. 1000 for safety net
actor_grad_clip: 1.0      # keep: actor gradient path through ODE is genuinely ill-conditioned
```

### 3. Always log pre-clip gradient norm (for monitoring)

Whether or not clipping is active, compute and log the total gradient norm so wandb plots remain useful.

## Files to Change

1. `agent/finetune/reinflow/train_sac_residual_flow_agent.py`
   - Read `critic_grad_clip` and `actor_grad_clip` from config (lines ~45-80)
   - Modify critic step (line 376): conditional clip, always log norm
   - Modify actor step (lines 396-398): use config value instead of hardcoded 1.0

2. `cfg/gym/finetune/hopper-v2/ft_sac_residual_flow_mlp.yaml`
   - Add `critic_grad_clip` and `actor_grad_clip` fields under `train:`

## What This Does NOT Change

- Actor clip stays at 1.0 by default (the ODE Jacobian path genuinely needs this)
- No architectural changes to the critic (that's a separate experiment)
- No weight decay added (orthogonal change, can be tested independently)
- Existing runs/checkpoints remain compatible (config fields have defaults)
