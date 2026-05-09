# SAC Auto-Entropy Tuning & SB3 Comparison

## Changes Made

### 1. Auto-entropy tuning (SB3-style)

**Files**: `train_sac_residual_flow_agent.py`, `ft_sac_residual_flow_mlp.yaml`

Added a learnable `log_alpha` with its own Adam optimizer. After each actor update:

```
alpha_loss = -(log_alpha * (log_prob_mean + target_entropy))
```

Gradient w.r.t. `log_alpha` is `-(log_prob + H_target)` -- independent of current alpha value.
This matches SB3 exactly (Haarnoja et al. 2018, eq. 18).

Config defaults:
- `auto_entropy_tuning: true`
- `init_alpha: 0.1` (starting value)
- `target_entropy: -12` (-action_dim * act_steps = -3 * 4)
- `alpha_lr: 3e-4` (same as SB3)

The previous codebase had a `loss_temperature` variant that multiplied `alpha` (not `log_alpha`)
by the detached log_prob, introducing an extra factor of alpha in the gradient. This caused
alpha to drop too fast when large and get stuck when small. The new implementation avoids this.

Checkpointing: `log_alpha` and `alpha_optimizer` are saved/restored on resume.

### 2. Removed sigma_ent_loss

**File**: `sac_residual_flow.py` (`loss_actor`)

Removed `sigma_ent_loss = -sigma_entropy_weight * sigma.log().sum().mean()`.

This was redundant with `entropy_loss = alpha * log_prob.mean()` -- both reduce to
`-c * log(sigma)` in the gradient w.r.t. sigma. The entropy loss is stronger
(alpha=0.1 vs sigma_entropy_weight=0.01), so sigma_ent_loss was contributing <10%
of an already-redundant signal.

With auto-entropy tuning, alpha itself adapts, making a separate sigma regularizer
both unnecessary and potentially harmful (it would fight the adaptive alpha).

`loss_actor` now also returns `log_prob_mean` in the info dict (needed by the alpha loss).

---

## Full SAC Comparison vs Stable Baselines 3

### Components that match SB3

| Component | Implementation | SB3 |
|-----------|---------------|-----|
| Alpha loss | `-(log_alpha * (logp + H).detach())` | Same |
| Alpha optimizer | Adam, lr=3e-4 | Same |
| Actor loss | `-min(Q1, Q2) + alpha * log_prob` | Same structure |
| Double Q critic | MSE on both Q1, Q2 vs target | Same |
| Target critic | EMA with tau=0.005 | Same (SB3 default: 0.005) |
| Replay buffer | FIFO, 1M capacity | Same |
| Batch size | 256 | Same |
| Gamma | 0.99 | Same |
| Critic LR | 3e-4 | Same |

### Intentional differences

**Target entropy: -12 vs -3.** SB3 uses `-prod(action_space.shape) = -3` for Hopper.
We use `-action_dim * act_steps = -12` because the policy outputs a 4-step action chunk
(shape [4, 3] = 12 dimensions). The entropy is summed over the full chunk.

**Critic target: hard Q (no entropy).** SB3 subtracts `alpha * log_pi(a'|s')` from the
Q-target (soft Bellman). We omit this because our `log_prob` is only the noise-layer
density `log N(eps; 0, sigma)`, NOT the true policy log-density `log pi(a|s)` (which
would require the full ODE logdet). Adding a partial quantity to the Bellman target
doesn't recover soft Q-learning -- it introduces a sigma-dependent bias with no
theoretical grounding. Hard-Q + entropy-regularized actor is a valid algorithm variant.

**Actor LR: 3e-5 (10x lower).** The actor gradient path goes through a K-step ODE
solver with Jacobian computations. This path is genuinely ill-conditioned and requires
a lower learning rate for stability.

**Actor update ratio: 1:4 (vs SB3's 1:1).** Delayed actor updates are common in SAC
variants (TD3 uses 1:2). Our ODE-based actor is more expensive and noisier than a
Gaussian policy, so less frequent updates are justified.

**Actor grad clip: 1.0.** SB3 doesn't clip. We clip because the ODE gradient path
can produce large spikes (the Jacobian of a multi-step flow map can amplify gradients).

**Critic warmup: 5000 iters.** SB3 has none. We warm up because the base policy
already achieves reasonable performance, and early critic estimates are poor. This gives
the critic time to learn before the actor starts chasing noisy Q-gradients.

### Not implemented (deliberately)

**Entropy in critic target** -- as explained above, our log_prob is not the true
policy density. Adding it to the Bellman target would be theoretically unmotivated.

---

## KL Loss: How It's Computed

### Per-step exact KL (`perstep_exact`, `perstep_exact_ref`)

At each ODE step k (k=0,1,2 for K=4 steps):

1. Run combined ODE: `a_{k+1} = a_k + (v_base + v_res)(a_k, t_k) * dt`
2. FP-invert one step of the reference map to find `a_inv` such that
   `a_inv + v_ref(a_inv, t_k) * dt = a_{k+1}`
3. Compute Jacobians at both points:
   - `J_ref` at `a_inv` (reference velocity Jacobian)
   - `J_comb` at `a_k` (combined velocity Jacobian)
4. Per-step KL contribution: `logdet(I + J_ref * dt) - logdet(I + J_comb * dt)`
5. Sum over K-1 steps

The total KL is `sum_k [logdet_ref_k - logdet_combined_k]`.
This equals `log p_theta - log p_base` by the change-of-variables formula.

For `_ref` modes, the reference is `v_base + ref_v_res` (EMA copy of v_res).
For non-ref modes, the reference is just `v_base` (frozen pretrained policy).

### Per-step Hutchinson KL (`perstep_hutchinson`, `perstep_hutchinson_ref`)

Same structure, but replaces `slogdet(I + J*dt)` with the Hutchinson trace estimator:
`logdet(I + J*dt) ~ tr(J) * dt`, estimated via `v^T J v` with Rademacher vectors.
Uses `hutchinson_samples` random probes (default: 4) per step.

### How KL enters the actor loss

```python
kl_loss = kl_weight * kl.mean()     # kl_weight = 0.05
total = sac_loss + kl_loss + entropy_loss
```

The KL loss is a weighted mean over the batch. With `kl_weight=0.05`, a raw KL of -3
contributes `loss_kl = 0.05 * (-3) = -0.15` to the total actor loss.

### KL regularizing power in previous runs (WandB data)

Late-training loss decomposition (steps 150k-200k):

```
Run                  loss_sac    loss_ent   loss_kl   |KL/SAC|
run15 exact*          -785.9      0.577     -0.145     0.02%
run16 hutch*          -797.8      0.576     -0.177     0.02%
run17 exact_ref*      -824.7      0.577      0.037     0.00%
run18 hutch_ref*      -827.6      0.577      0.021     0.00%
run5  exact           -824.2      0.575     -0.149     0.02%
run6  hutchinson      -854.9      0.575     -0.398     0.05%
run9  exact_ref       -857.9      0.575      0.350     0.04%
run10 hutch_ref       -860.2      0.578     -0.327     0.04%
A2   baseline         -885.3      0.548      0.000      --
```

**KL is 0.02-0.05% of the SAC loss magnitude.** The Q-gradient dominates at ~800;
KL contributes 0.02-0.4. The actor optimizer effectively ignores KL.

**sigma = 0.15 (saturated at sigma_max) in every run.** The fixed alpha=0.1 pushes
sigma unconditionally upward via `-alpha/sigma` gradient, overwhelming the only
downward force (Q-gradient through the noisy action).

**With auto-entropy tuning**, alpha will decay toward a value where `E[log_prob] ~
target_entropy = -12`. As alpha shrinks, the entropy pressure on sigma weakens, sigma
finds an equilibrium, and KL's relative contribution to the actor loss grows. This is
when KL should start differentiating runs.

---

## Lagrangian Dual Variable for Adaptive KL Weight

### 3. KL Lagrange multiplier (MPO-style)

**Files**: `train_sac_residual_flow_agent.py`, `ft_sac_residual_flow_mlp.yaml`

**Problem**: With a fixed `kl_weight`, the KL loss is 0.02-0.05% of the SAC loss
magnitude. The Q-gradient dominates the actor update, and KL has no meaningful
regularizing effect. Even worse, KL needs to be effective *early* in training (when
alpha is still high and Q-values are growing) to prevent catastrophic forgetting —
auto-entropy tuning alone doesn't solve this.

**Solution**: Replace the fixed `kl_weight` with a learnable Lagrange multiplier `eta`
(optimized in log-space) that enforces a soft constraint `E[KL] <= epsilon`. This is
the standard approach from MPO/V-MPO (Abdolmaleki et al. 2018).

The actor loss becomes:

```
actor_loss = -Q_min + eta.detach() * KL + alpha * log_prob
```

After each actor update, the dual variable is updated:

```
eta_loss = eta * (epsilon - KL.detach().mean())
```

- When `KL > epsilon`: `eta_loss < 0`, gradient pushes `log_eta` up → eta increases →
  stronger KL penalty next step
- When `KL < epsilon`: `eta_loss > 0`, gradient pushes `log_eta` down → eta decreases →
  weaker KL penalty next step

The log-space parameterization (`eta = exp(log_eta)`) ensures eta stays positive. The
`.detach()` calls are critical: `eta` treats KL as a fixed signal (dual problem), while
the actor treats `eta` as a fixed weight (primal problem).

**Why this works when fixed kl_weight doesn't**: eta automatically scales to match the
Q-gradient magnitude. If Q-values are ~800 and KL is ~3, eta will rise until
`eta * KL ≈ O(Q)` — the optimizer has no choice but to attend to the KL term. A fixed
weight of 0.05 produces `0.05 * 3 = 0.15` against a Q-gradient of 800 — invisible.

Config defaults:
- `kl_lagrangian: false` (off by default, backwards compatible)
- `kl_target_epsilon: 1.0` (KL budget; start with 1.0-5.0)
- `kl_init_eta: 1.0` (initial multiplier)
- `kl_eta_lr: 1e-3` (dual variable learning rate)

Checkpointing: `log_eta` and `eta_optimizer` are saved/restored on resume.

WandB logging: `actor/eta` (current multiplier) and `actor/eta_loss` (dual loss) are
logged automatically via `last_actor_info`.

### Usage

To enable, set in the experiment config:
```yaml
train:
  kl_lagrangian: true
  kl_target_epsilon: 1.0    # tune this: smaller = tighter KL constraint
  kl_init_eta: 1.0
  kl_eta_lr: 1.0e-3
  kl_mode: perstep_exact     # or any KL mode that produces kl_mean
```

The `kl_weight` field in the config is ignored when `kl_lagrangian: true`.
