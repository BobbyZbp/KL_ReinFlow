# KL Stabilization: Losses, Gradients, and NaN Analysis

## Setup

We fine-tune a frozen flow-matching base policy `v_base` with a small trainable residual `v_res`, using SAC. The forward ODE runs K=4 Euler steps at dt=0.25:

```
a^0 ~ N(0, I)                                          # initial noise
a^{k+1} = a^k + (v_base(a^k, t_k) + v_res(a^k, t_k)) * dt,   k = 0..K-2   (deterministic)
a^K = a^{K-1} + v_combined(a^{K-1}, t_{K-1}) * dt + sigma * eps   (noisy last step)
```

The executed action is `a^K`. The critic evaluates `Q(s, a^K)`. Everything below describes the **actor loss** — how gradients from different loss terms reach the trainable parameters (`v_res`, `sigma_head`).

---

## Shared loss components (all five methods)

### SAC Q-value loss

```
sac_loss = -min(Q1(s, a^K), Q2(s, a^K))
```

Gradient path: `sac_loss → a^K → a^{K-1} → ODE steps → v_res`. This is the standard policy gradient via reparameterization. Well-conditioned — just backprop through the Euler forward pass. This is **Path 1**.

### SAC entropy (alpha * log_prob)

```
log_prob = sum(-0.5 * eps^2 - log(sigma) - 0.5 * log(2*pi))
entropy_loss = alpha * log_prob.mean()
```

With `alpha = 0.1`. The `eps` is sampled noise (detached from parameters), so `eps^2` contributes no gradient. The gradient flows only through `-log(sigma)` → `sigma_head` parameters. This keeps the policy stochastic and prevents Q-value overestimation. **Present in all five methods.**

### Sigma entropy bonus

```
sigma_ent_loss = -sigma_entropy_weight * mean(sum(log(sigma)))
```

With `sigma_entropy_weight = 0.1`. Encourages larger sigma for exploration. Gradient flows through `sigma_head` only. **Present in all five methods.**

---

## The NaN problem: what causes it

### Root cause

To compute `KL(p_theta || p_base)`, we need the log-densities of both the combined policy and the base policy. For a deterministic ODE, the change-of-variables formula gives:

```
log p(a^{K-1}) = log p_0(a^0) - sum_{k=0}^{K-2} log|det(I + J_k * dt)|
```

where `J_k = d(v(a^k, t_k)) / d(a^k)` is the Jacobian of the velocity field.

Computing `log|det(M)|` where `M = I + J*dt` uses `torch.linalg.slogdet`. The forward pass is fine — it's O(D^3) but numerically stable. The problem is the **backward pass**: the gradient of `slogdet(M)` with respect to `M` is `M^{-T}` (the inverse transpose). When `M` is near-singular (i.e., `J*dt` has an eigenvalue near -1), the inverse explodes, producing NaN gradients that propagate through the entire network.

### Three gradient paths from KL loss to v_res

When we compute `kl = log_p_theta - log_p_base` and backpropagate:

**Path 1** (safe): `kl → log_p_theta → a^0 (noise, no params)` — log_p0 depends on initial noise, no v_res gradient.

**Path 2** (dangerous, theta-side): `kl → log_p_theta → slogdet(I + J_combined * dt) → jacrev(v_combined) → v_res`.
The slogdet backward computes `(I + J_combined * dt)^{-1}`. If the combined velocity's Jacobian makes this near-singular, gradients explode.

**Path 3a** (benign, base-side): `kl → log_p_base → log_p0(a^0_base) → fixed-point chain → a^{K-1} → v_res`.
Gradient through the initial noise density, propagated back through the trajectory. Mild.

**Path 3b** (dangerous, base-side — **DISCOVERED DURING EXPERIMENTS**): `kl → log_p_base → slogdet(I + J_base * dt) → a^k_base → fixed-point chain → a^{K-1} → v_res`.
The `backward_base_logprob` method also calls `slogdet` on `I + J_base * dt` (lines 479-483 of sac_residual_flow.py). The trajectory points `a^k_base` are differentiable w.r.t. `a^{K-1}` through the one-step refinement chain (lines 460-466). So backprop through this slogdet requires `(I + J_base * dt)^{-1}` — the **same singular-inverse problem**, just on the base policy's Jacobian.

### Discovery timeline

Initially we thought only Path 2 was dangerous. We implemented Hutchinson (avoids Path 2 by replacing slogdet with JVP) and detach_logdet (avoids Path 2 by detaching the logdet). Both still died:
- Hutchinson: NaN at iter 6973
- detach_logdet: NaN at iter 6329

Investigation revealed Path 3b. The fix: **detach `log_p_base` entirely** for modes that try to avoid slogdet gradients. KL value is still computed for monitoring, but `log_p_base.detach()` means zero gradient flows back through the base-side slogdet.

After the fix, both Hutchinson and detach_logdet survived 170k+ iterations with zero NaN.

---

## Method-by-method breakdown

### A — Baseline (`kl_mode=none`, `alpha=0.1`)

**Actor loss:**
```
total = sac_loss + entropy_loss + sigma_ent_loss
      = -Q(s, a^K) + 0.1 * log_prob + sigma_entropy_bonus
```

**What gets gradient:**
- `v_res` ← only from `sac_loss` via Path 1 (Q-value → action → ODE → v_res)
- `sigma_head` ← from `entropy_loss` (-log(sigma)) and `sigma_ent_loss` (log(sigma))

**KL involvement: None.** No KL computation at all. The policy is free to diverge arbitrarily from the base. The only constraint on v_res is implicit: the SAC Q-function rewards good actions, and the entropy term keeps exploration alive.

**Why it works:** With `alpha=0.1`, the entropy term prevents the critic from overestimating Q-values for deterministic actions. Without it (`alpha=0`), the critic diverges within ~900 actor iterations.

---

### B — Reward penalty (`kl_mode=reward_penalty`, `kl_reward_weight=0.05`)

**Actor loss (identical to A):**
```
total = sac_loss + entropy_loss + sigma_ent_loss
```

**But the reward stored in the replay buffer is modified during collection:**
```python
# At environment step time (no gradients):
with torch.no_grad():
    kl = compute_kl_for_reward(obs)       # full forward+backward, exact KL
r_modified = r_env - 0.05 * kl            # stored in replay buffer
```

The `compute_kl_for_reward` method runs the full `forward_with_logdet` (exact Jacobians + slogdet) and `backward_base_logprob` (fixed-point inversion + base slogdet) under `torch.no_grad()`. It draws fresh noise (not the noise used for the executed action) and computes `E_z[KL(z)]`.

**What gets gradient:**
- `v_res` ← only from `sac_loss` via Path 1, but Q-function was trained on KL-penalized rewards
- `sigma_head` ← same as A

**KL involvement: Yes, but indirectly.** KL never appears in the actor loss or its gradients. Instead, the critic learns `Q(s, a) = E[sum r_env - 0.05 * KL]`. The actor maximizes this Q, which implicitly penalizes high-KL actions. The KL signal is **lagged**: it enters the replay buffer → trains the critic over many batches → actor sees it through `-Q(s, a^K)`.

**Why it's the slowest (0.33s/step vs 0.31s):** Every environment step computes full exact KL (Jacobians + slogdet + fixed-point inversion) under no_grad. This is cheap in GPU compute but adds ~6% wall-clock overhead per step. It processes 146k iters in 11h vs 180k for A.

**Gradient safety: Fully safe.** All KL computation is under `torch.no_grad()`. No dangerous paths. The actor's backward pass is identical to A.

---

### C — Hutchinson trace estimator (`kl_mode=hutchinson`, `kl_weight=0.05`)

**Actor loss:**
```
total = sac_loss + 0.05 * KL_hutchinson + entropy_loss + sigma_ent_loss
```

where `KL_hutchinson = log_p_theta_hutch - log_p_base.detach()` (after the fix).

**How log_p_theta is computed (Hutchinson):**

Instead of computing `log|det(I + J*dt)|` via full Jacobian + slogdet, we approximate:

```
log|det(I + J*dt)| ≈ tr(J) * dt
```

This is the first-order Taylor expansion (`log det(I + M) ≈ tr(M)` for small `||M||`). The trace is estimated via Hutchinson's trick:

```python
v_probe ~ Rademacher(±1)
Jv = jvp(velocity_fn, (a,), (v_probe,))[1]     # Jacobian-vector product, O(D) cost
tr(J) ≈ v_probe^T * Jv
```

This requires only a JVP (forward-mode autodiff), not the full D×D Jacobian. The backward pass through a JVP is clean — it's just a directional derivative, no matrix inverse anywhere.

**Why Hutchinson still involves Path 3b (before the fix):**

The Hutchinson estimator only changes how `log_p_theta` is computed — the theta side (Path 2). But KL = `log_p_theta - log_p_base`, and `log_p_base` is computed by `backward_base_logprob` regardless of how we handled the theta side. That method calls `slogdet(I + J_base * dt)` on trajectory points `a^k_base` that are differentiable w.r.t. `a^{K-1}`. So without any fix, backprop through KL still requires `(I + J_base * dt)^{-1}` via Path 3b. The Hutchinson trick fixes one half of the KL gradient (theta side) but leaves the other half (base side) untouched.

This is why Hutchinson NaN'd at iter 6973 in the first round of experiments — the NaN came from the base-side slogdet, not the theta-side trace estimator.

**The Path 3b fix:** `log_p_base` is computed with `torch.no_grad()` and then `.detach()`ed. This means:
- KL **value** = `log_p_theta_hutch - log_p_base` (correct for monitoring)
- KL **gradient** = only through `log_p_theta_hutch` (Hutchinson JVP, safe)
- No gradient through `backward_base_logprob` at all (Path 3b eliminated)

**What gets gradient (after the fix):**
- `v_res` ← from `sac_loss` (Path 1) **and** from `kl_weight * KL` through `log_p_theta_hutch`. The gradient flows: `KL → log_p_theta → sum(tr_est * dt) → JVP → velocity_fn → v_res`. This is a **direct actor gradient from KL** — the only method besides exact that has this.
- `sigma_head` ← from entropy and sigma_ent terms

**KL involvement: Yes, directly in the actor loss.** The KL term's gradient through `log_p_theta` pushes `v_res` to keep the combined policy's density close to the base. The `log_p_base` is detached, so the gradient says "make log_p_theta smaller" (i.e., keep theta's density low at points where it would otherwise concentrate mass away from the base).

**Approximation quality:** With `||J*dt||` potentially large (especially at later ODE steps near t→1), the first-order approximation `log det(I+M) ≈ tr(M)` is biased. The correction terms are `-tr(M^2)/2 + tr(M^3)/3 - ...`. With only 1 Hutchinson sample, there's also high variance. This likely explains the reward dip to ~887 at iter ~129k (the noisy trace estimate occasionally gives a bad gradient direction that destabilizes the policy temporarily).

---

### D — Detach logdet (`kl_mode=detach_logdet`, `kl_weight=0.05`)

**Actor loss:**
```
total = sac_loss + 0.05 * KL_detach + entropy_loss + sigma_ent_loss
```

where `KL_detach = log_p_theta_detach - log_p_base.detach()`.

**How log_p_theta is computed:**

The exact `log|det(I + J*dt)|` is computed via full Jacobian + slogdet, but under `torch.no_grad()`:

```python
with torch.no_grad():
    J = vmap(jacrev(velocity_fn))(a, t, cond)
    M = I + J * dt
    _, logabsdet = torch.linalg.slogdet(M)
sum_logdet += logabsdet    # exact value, but no gradient through it
log_p_theta = log_p0 - sum_logdet
```

So `log_p_theta` has exact numerical value, but `sum_logdet` is a constant w.r.t. parameters. And `log_p0 = log N(a^0; 0, I)` depends on initial noise `a^0`, which is sampled (no parameter dependence).

**What gets gradient:**
- `v_res` ← only from `sac_loss` (Path 1). **The KL term contributes zero gradient to v_res.** Here's why:
  - `log_p_theta = log_p0 - sum_logdet_detached`. `log_p0` depends on `a^0` (sampled noise, no params). `sum_logdet` is detached. So `∂log_p_theta/∂θ = 0`.
  - `log_p_base` is also detached.
  - Therefore `∂KL/∂θ = 0`. The KL loss adds a **constant** to the total loss — it shifts the loss value but not the gradient.
- `sigma_head` ← from entropy and sigma_ent terms

**KL involvement: No.** Despite computing and adding `0.05 * KL` to the loss, the gradient contribution is exactly zero. This method is functionally identical to A (baseline) in terms of what the optimizer sees. The KL value appears in logs and wandb, which is useful for monitoring, but it has no training effect.

**Why it performs worse than A:** It's effectively the same method but with different seed (45 vs 42) and the computational overhead of running slogdet (even under no_grad). The ~2800 late-stage reward vs A's ~3100 is seed variance, not a methodological difference.

---

### E — v_res L2 penalty (`kl_mode=vres_l2`, `vres_l2_weight=0.01`)

**Actor loss:**
```
total = sac_loss + 0.01 * mean(||v_res(a^k, t_k)||^2) + entropy_loss + sigma_ent_loss
```

**How the penalty is computed:**

```python
penalty = 0
for k in range(K-1):          # k = 0, 1, 2 (3 ODE steps)
    v_k = v_res(traj_a[k].detach(), traj_t[k], cond)
    penalty += mean(||v_k||^2)
penalty /= (K - 1)
```

The trajectory points `traj_a[k]` come from `sample_action(return_trajectory=True)` and are **detached**. So the gradient only flows through `v_res` at each evaluation point, not back through the ODE that produced those points.

**What gets gradient:**
- `v_res` ← from `sac_loss` (Path 1) **and** from the L2 penalty. The L2 gradient is `∂||v_res(a, t)||^2 / ∂θ = 2 * v_res * ∂v_res/∂θ`. This pushes all v_res outputs toward zero uniformly.
- `sigma_head` ← from entropy and sigma_ent terms

**KL involvement: No (proxy only).** `||v_res||^2` is a proxy for KL: since `v_res = 0` ⟹ `p_theta = p_base` ⟹ `KL = 0`, small v_res implies small KL. But it's not KL itself. The penalty treats all directions equally — it can't distinguish "v_res moves probability mass in a useful direction" from "v_res moves mass in a harmful direction." It's a blunt instrument.

**Weight analysis:** At `vres_l2_weight=0.01` with `||v_res||^2` values that start near zero (zero-initialized residual), this penalty is negligible compared to `sac_loss` (magnitude ~900). The method is effectively identical to A until v_res grows large enough for 0.01 * ||v_res||^2 to matter. This explains why E tracks A closely but with seed-dependent variance.

---

## Summary table

| Method | KL in actor loss? | KL gradient to v_res? | KL in reward? | Dangerous paths? | What actually regularizes v_res? |
|--------|-------------------|----------------------|---------------|-----------------|--------------------------------|
| A (none) | No | No | No | None | Only Q-values |
| B (reward_penalty) | No | No (indirectly via Q) | Yes | None (no_grad) | Q-values trained on r - 0.05*KL |
| C (hutchinson) | Yes | Yes (via JVP trace) | No | None (Path 3b fixed) | Q-values + direct KL gradient through Hutchinson trace |
| D (detach_logdet) | Numerically yes, gradient no | No (zero gradient) | No | None (all detached) | Only Q-values (same as A) |
| E (vres_l2) | No | No | No | None | Q-values + 0.01 * ||v_res||^2 penalty |

### Which methods actually use KL for training?

**Only B and C.**

- **B** uses exact KL to modify the reward signal. The actor never differentiates through KL, but the critic learns that high-KL actions have lower value. The KL signal is real but lagged (replay buffer → critic → actor).

- **C** uses approximate KL (Hutchinson trace) directly in the actor loss. The gradient `∂KL/∂θ` flows through the trace estimator's JVP, giving a direct (if noisy) signal to keep `log_p_theta` close to `log_p_base`. This is the only method where the actor's gradient explicitly contains a KL term.

- **D** computes exact KL but contributes zero gradient. It's a monitoring tool, not a training signal.

- **E** uses `||v_res||^2` as a proxy for KL, but this is not KL — it penalizes the magnitude of the residual uniformly regardless of direction.

### How we solved NaN

1. **SAC entropy (`alpha=0.1`)** — the critical first fix. Without it, Q-value overestimation causes critic divergence regardless of KL mode.

2. **Detaching `log_p_base`** — the key discovery. Both the theta-side (`slogdet` of combined Jacobian) and the base-side (`slogdet` of base Jacobian in `backward_base_logprob`) have the same failure mode: `slogdet` backward requires `M^{-1}`, which explodes when `M = I + J*dt` is near-singular. For Hutchinson and detach_logdet modes, we detach `log_p_base` entirely (`torch.no_grad()` around `backward_base_logprob`), eliminating Path 3b. The KL value is still correct for logging; only the gradient through the base density is dropped.

After both fixes, all five methods survived 145k–180k iterations with zero NaN.
