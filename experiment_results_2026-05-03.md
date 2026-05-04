# KL-ReinFlow Training Stability Experiments (2026-05-03)

## Goal

Determine why training diverges (NaN) when actor updates begin, and whether the exact KL regularizer is the cause.

## Setup

- **Environment**: Hopper-medium-v2 (D4RL), 40 parallel envs (SyncVectorEnv)
- **Base policy**: Pretrained 1-step ReFlow (reward ~1260)
- **Architecture**: Residual velocity v_res (24K params) + frozen v_base (553K params), twin Q-critic (279K params), learned sigma noise head (6K params)
- **GPU**: A5500 24GB (SLURM interactive session)

## Config differences vs release branch

| Parameter | Release | Ours (KL run) | Ours (no-KL run) |
|---|---|---|---|
| `alpha` (SAC entropy) | **0.1** | 0.0 | 0.0 |
| `n_explore_steps` | **5000** | 1000 | 1000 |
| `actor_replay_ratio` | **128** (freq=2) | 64 (freq=4) | 64 (freq=4) |
| `critic_warmup_iters` | **0** | 5000 | 5000 |
| `kl_weight` | 0 (commented out) | **0.05** | **0.0** |
| `jac_weight` | 0 (commented out) | **0.01** | **0.0** |
| `sigma_entropy_weight` | N/A | 0.01 | 0.01 |
| Actor loss function | `sample_action` + entropy | `compute_kl_and_action` + KL | `compute_kl_and_action` + KL (zero-weighted) |

## Training timeline

Both runs follow the same schedule:
- **Iters 0-999**: Random exploration (no training)
- **Iters 1000-5999**: Critic-only training (actor frozen, base policy collects data)
- **Iters 6000+**: Both critic and actor training (actor every 4th iter)

## Results

### KL run (kl_weight=0.05, jac_weight=0.01)

| Phase | Iters | Critic Loss | Reward | Status |
|---|---|---|---|---|
| Critic warmup | 1001 | 24.9 | 18 (random) | OK |
| Critic warmup | 2000 | 17.0 | 1261 | OK |
| Critic warmup | 3000 | 33.0 | 1275 | OK (past old async crash zone) |
| Critic warmup | 5000 | ~100-260 | ~1248 | OK, loss growing |
| Actor starts | 6000 | 93.6 | 1264 | Actor loss = -214.9 |
| Last healthy | 6148 | 65.0 | 1269 | Actor loss = -221.8 |
| **First NaN** | **6149** | **NaN** | 1271 | **Critic NaN first, then actor** |
| Terminal | 34599 | NaN | 144 | Irrecoverable |

**Actor survived 149 iterations before NaN.**

### No-KL run (kl_weight=0.0, jac_weight=0.0)

| Phase | Iters | Critic Loss | Reward | Status |
|---|---|---|---|---|
| Critic warmup | 1001 | 25.2 | 18 (random) | OK |
| Critic warmup | 2000 | 19.3 | 1288 | OK |
| Critic warmup | 3000 | 50.5 | 1213 | OK |
| Critic warmup | 5000 | ~100-260 | ~1245 | OK, loss growing |
| Actor starts | 6000 | 117.8 | 1275 | Actor loss = -211.3 |
| Stable actor | 6500 | ~100-400 | ~1260 | Actor loss = -230 |
| Last healthy | 6920 | 233.4 | 1260 | Actor loss = -242.6 |
| **First NaN** | **6921** | **NaN** | 1260 | **Critic NaN first, then actor** |
| Terminal | 34353 | NaN | 149 | Irrecoverable |

**Actor survived 920 iterations before NaN (5x longer than KL run).**

## Key finding: critic diverges first in both runs

In both cases, the **critic goes NaN before the actor**:
```
# No-KL run, iter 6920-6921:
6920: actor -242.6323 | critic 233.4306 | reward 1259.5  # healthy
6921: actor -242.6323 | critic      nan | reward 1259.5  # critic NaN, actor still fine
```

The NaN then propagates: critic NaN → actor NaN → policy outputs garbage → reward collapses (1260 → 144).

## Root cause analysis

### Why critic diverges: missing SAC entropy

The release branch uses **alpha=0.1** with a standard SAC entropy bonus in `loss_actor`:
```python
log_prob = (-0.5*eps^2 - log(sigma) - 0.5*log(2*pi)).sum(dims)
entropy_loss = 0.1 * log_prob.mean()
total = sac_loss + entropy_loss
```

Our branch replaced this with sigma entropy at **weight=0.01**:
```python
sigma_ent_loss = -0.01 * sigma.log().sum(dims).mean()
total = sac_loss + kl_loss + jac_loss + sigma_ent_loss
```

Since `eps` is detached noise (`torch.randn_like`), the `eps^2` term in log_prob has zero gradient. **Both losses produce the same gradient direction** (push sigma up for exploration). The difference is purely the weight: **0.1 vs 0.01** (10x weaker).

Neither branch subtracts entropy from the critic target (non-standard SAC). The only entropy mechanism is in the actor loss, which keeps the policy stochastic and prevents the critic from overfitting to a narrow action distribution.

### Why KL makes it worse

The KL run dies 5x faster (149 vs 920 actor iters). The `compute_kl_and_action` function involves backward fixed-point inversion (30 iterations) to recover base-policy noise from the current policy's actions. This computation:
1. Is numerically stiff (error grows as v_res diverges from zero)
2. Can produce NaN in kl/jac_reg values
3. Even with `kl_weight=0.0`, the expression `0.0 * NaN = NaN` in IEEE 754, which would poison the total loss

### Other contributing factors

1. **`n_explore_steps: 1000` vs 5000**: Our buffer has 40K random transitions vs 200K in release. Less diverse initial data.
2. **`critic_warmup_iters: 5000`**: The critic trains 5000 steps alone before the actor starts. By iter 6000, Q-values are already large and potentially overestimated. Release starts actor+critic together.
3. **`actor_replay_ratio: 64` vs 128**: Actor updates every 4 iters instead of every 2. Fewer actor updates means slower entropy response to critic overestimation.

## Critic loss growth pattern (both runs)

The critic loss steadily increases throughout training, even during critic-only warmup:
```
Iter 1000: ~25
Iter 2000: ~17-19
Iter 3000: ~33-50
Iter 4000: ~45-49
Iter 5000: ~100-260 (high variance)
Iter 6000: ~94-118
Iter 6500: ~100-400 (spikes)
Iter 6900: NaN
```

This growth pattern indicates Q-value overestimation accumulating over time, consistent with the missing entropy regularization.

## Previous runs (v1, v2) — AsyncVectorEnv crashes

Before the SyncVectorEnv fix, all runs crashed at iter ~3100 with `BrokenPipeError` from `async_vector_env.py`. This was caused by MuJoCo NaN states in child processes killing the async workers. The fix in `agent/finetune/train_agent.py` (line 85: `asynchronous=cfg.env.get("asynchronous", False)`) resolved this completely.

## Next steps

1. **Restore SAC entropy**: Set `alpha=0.1` (or tune) and add `alpha * log_prob` back to the actor loss alongside KL terms
2. **Match release explore schedule**: Consider `n_explore_steps=5000` and removing critic warmup
3. **Guard against KL NaN**: Use `torch.nan_to_num` on kl/jac before multiplying by weight, or check finiteness before adding to loss
4. **Validate**: Run release config unchanged to confirm it trains stably as reported
