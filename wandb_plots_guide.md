# Understanding wandb Training Plots

Guide for interpreting the training metrics logged during SAC residual flow training.

## Episode metrics (`episode/reward_mean|std|min|max`)

Computed over `completed_ep_rewards`, a `deque(maxlen=200)` storing the **total return of each finished episode** (sum of all per-step rewards within one episode). Every time an environment signals `done`, the accumulated reward for that episode is appended. The mean/std/min/max are computed over this 200-episode sliding window.

**Why `reward_max` surges to >5000**: Hopper episodes have variable length. A typical good episode might last ~300 steps at ~3 reward/step = ~900 total return. Occasionally the policy finds a gait that survives much longer (1500+ steps), and that one episode accumulates a much larger total reward. Since `reward_max` reports the single best episode in the 200-episode window, one outlier creates a sharp spike that persists until it falls out of the window -- hence the periodic surge pattern.

## `reward/raw_mean` vs `episode/reward_mean`

- `reward/raw_mean` -- per-step instantaneous reward, averaged over `recent_raw_rewards` (another deque). Tells you "how good are individual transitions right now."
- `episode/reward_mean` -- total episodic return, averaged over the last 200 completed episodes. This is the metric that matters for evaluation.

They differ by roughly a factor of episode length.

## Critic metrics

### `critic/loss`

Standard SAC TD loss: `MSE(Q1, target) + MSE(Q2, target)` where:

```
target = r + gamma * (1 - done) * min(Q1_target, Q2_target)
```

Should decrease and stabilize. If it diverges (spikes, NaN), training is about to collapse.

### `critic/q_target_mean|std|min`

Computed over the **training batch** (256 transitions sampled from the replay buffer) in `loss_critic()`. `q_target_mean` is the average Bellman backup across the batch, `q_target_std` measures spread, `q_target_min` is the lowest target.

Useful for detecting critic divergence -- if `q_target_mean` explodes or goes NaN, the critic is unstable.

## Actor metrics

### `actor/loss_sac`

```
L_sac = -min(Q1, Q2).mean()
```

**This is negative** because Q-values are positive (they estimate future cumulative reward in Hopper). The negation makes the loss negative. The more the critic values the policy's actions, the more negative the loss. Gradient descent on a negative loss pushes the policy toward higher Q-values.

### `actor/q_mean`

`min(Q1(s,a), Q2(s,a)).mean()` where `a` is freshly sampled from the current policy (not from the replay buffer). Tells you how valuable the critic thinks the policy's current actions are. Should trend upward during successful training.

### `actor/sigma`

The exploration noise standard deviation from `SigmaHead`. Bounded by `[sigma_min=0.05, sigma_max=0.15]`.

**Why it keeps increasing**: the sigma entropy loss pushes it up:

```
L_sigma_ent = -sigma_ent_weight * sum(log(sigma)).mean()
```

with `sigma_ent_weight = 0.1`. The gradient is `-0.1/sigma`, always negative, always pushing sigma to grow. Sigma increases until it hits `sigma_max = 0.15` and plateaus.

Purpose: sigma controls Gaussian exploration noise added at the final ODE step. Maximizing its entropy prevents the policy from collapsing to deterministic actions too early.

### `actor/loss_entropy`

SAC entropy bonus:

```
L_entropy = alpha * (-0.5*eps^2 - log(sigma) - 0.5*log(2*pi)).sum().mean()
```

Only `-log(sigma)` has gradient (eps is detached). This is the standard SAC entropy regularizer.

## Loss components (full actor loss)

```
L_total = L_sac + L_kl + L_jac + L_sigma_ent + L_entropy + L_vres
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| `loss_sac` | `-min(Q1, Q2).mean()` | Policy gradient (maximize Q) |
| `loss_kl` | `kl_weight * kl.mean()` | Stay close to base policy |
| `loss_jac` | `jac_weight * jac_norm.mean()` | Jacobian Frobenius regularization |
| `loss_sigma_ent` | `-sigma_ent_weight * sum(log(sigma)).mean()` | Maximize exploration noise entropy |
| `loss_entropy` | `alpha * log_prob.mean()` | SAC entropy bonus |
| `loss_vres` | `vres_weight * vres_norm.mean()` | L2 regularization on residual velocity |

## Caps and bounds

| Quantity | Bound | Source |
|----------|-------|--------|
| sigma | [0.05, 0.15] | `SigmaHead` (sigma_min / sigma_max) |
| KL per step | [-1e4, 1e4] | `torch.clamp` in `compute_perstep_kl_and_action()` |
| Actions | [-1, 1] | Clipped before environment step |
| `loss_sigma_ent` | Implicitly bounded | sigma in [0.05, 0.15] => log(sigma) in [-3.0, -1.9], for D=3: loss in ~[-0.9, -0.57] |
| `loss_sac` | No explicit cap | Bounded indirectly by critic magnitude |
| Q-values | No explicit cap | Can diverge if critic is unstable |

## Which plots matter most (priority order)

1. **`episode/reward_mean`** -- primary training signal. Is the policy improving?
2. **`critic/loss`** -- should decrease and stabilize. Divergence = imminent collapse.
3. **`actor/loss_sac`** -- should become more negative as Q-values grow. Sudden jumps = instability.
4. **`kl/perstep_kl_mean`** -- how much the policy has drifted from base. Should stay moderate; explosion = KL weight too low.
5. **`actor/sigma`** -- should climb toward 0.15 and plateau. Stuck at 0.05 = sigma_ent_weight too low.
6. **`critic/q_target_mean`** -- should grow with reward. Divergence from `episode/reward_mean` = value estimation problem.

## Code references

- Episode reward tracking: `train_sac_residual_flow_agent.py` -- `completed_ep_rewards = deque(maxlen=200)`
- Critic loss: `sac_residual_flow.py:loss_critic()` (line ~783)
- Actor loss: `sac_residual_flow.py:loss_actor()` (line ~803)
- Per-step KL: `sac_residual_flow.py:compute_perstep_kl_and_action()` (line ~554)
- SigmaHead bounds: `sac_residual_flow.py:SigmaHead` class
