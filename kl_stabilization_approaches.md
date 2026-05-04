# KL Stabilization Approaches for Flow-Matching SAC

## Problem

Training a residual flow-matching policy with SAC + exact deterministic KL diverges (NaN) because the gradient path through `slogdet` backward requires computing `(I + J*dt)^{-1}`, which is ill-conditioned when the Jacobian `J = d(v_base + v_res)/da` makes `I + J*dt` near-singular.

### Gradient anatomy of `compute_kl_and_action`

Three paths from `total.backward()` to `v_res` parameters:

1. **Path 1 (clean)**: `sac_loss → a_K → a_{K-1} → ODE steps → v_res`. Just the forward pass — well-conditioned.
2. **Path 2 (DANGEROUS)**: `kl_loss → log_p_theta → sum_logdet → slogdet(I+J*dt) → jacrev(v_combined) → v_res`. This is second-order: `slogdet` backward requires `(I+J*dt)^{-1}`, which blows up when near-singular.
3. **Path 3 (complex)**: `kl_loss → log_p_base → log_p0(a^0_base) → backward FP chain → a_{K-1} → v_res`. Stiff fixed-point inversion with 30 iterations.

The current implementation computes exact KL = log p_theta - log p_base, where:
- `log p_theta = log p_0(z) - sum_k log|det(I + J_combined * dt)|` (forward change of variables)
- `log p_base` = backward fixed-point inversion to recover base noise, then forward log-det along base trajectory

## Prerequisite (all approaches)

Restore SAC entropy: `alpha = 0.1` (matching release branch). Without this, Q-value overestimation causes NaN even without KL. The sigma_entropy at weight 0.1 is equivalent in gradient direction but the release-style `alpha * log_prob` also helps stabilize the critic target indirectly.

---

## Approaches (ranked by practicality)

### 1. KL as reward penalty (no actor gradient from KL)

**Idea**: Compute KL with `torch.no_grad()`, subtract it from the reward in the replay buffer. The actor never sees KL gradients — it only sees the modified reward through the critic.

**Implementation**:
- At data collection time, compute `kl = log_p_theta - log_p_base` (detached) for each transition
- Store `r_modified = r - lambda * kl` in the replay buffer
- Actor loss is pure SAC: `-Q(s, a) + alpha * log_prob`

**Pros**: Zero risk of NaN from KL gradients. KL signal reaches actor indirectly through Q-values. Simple to implement.

**Cons**: KL signal is lagged (enters via replay buffer, then critic, then actor). May be too weak to prevent large deviations early on. Requires computing KL at collection time (adds latency to env step). The KL is computed at a fresh noise draw (not the executed trajectory's `a^{K-1}`) because computing `log_p_theta` at an arbitrary point requires backward-inverting the combined ODE — the same stiff fixed-point problem. This is unbiased (estimating `E_z[KL]`) but adds per-sample variance.

**Compute cost**: Same KL computation but only at collection time (1x per env step), not at training time (many times per batch). No Jacobian in actor backward pass.

---

### 2. Hutchinson trace estimator (JVP-based, no slogdet)

**Idea**: Replace `log|det(I + J*dt)|` with `tr(J) * dt` (first-order approximation valid when `||J*dt|| < 1`). The trace can be estimated via Hutchinson's trick: `tr(J) ≈ v^T J v` where `v ~ Rademacher`. This only requires a JVP (Jacobian-vector product), not the full Jacobian or slogdet.

**Implementation**:
```python
v = torch.randint(0, 2, (B, D)) * 2 - 1  # Rademacher
Jv = jvp(velocity_fn, (a,), (v,))[1]      # JVP, O(D) cost
tr_est = (v * Jv).sum(dim=-1)              # v^T J v ≈ tr(J)
logdet_approx = tr_est * dt
```

**Pros**: O(D) per step instead of O(D^3) for slogdet. No matrix inverse in backward pass — gradient is through the JVP, which is just a directional derivative. Well-established in neural ODE literature (FFJORD).

**Cons**: Biased when `||J*dt||` is large (higher-order terms in `log det(I+M) = tr(M) - tr(M^2)/2 + ...`). With K=4, dt=0.25, and D=12, the approximation quality depends on the Jacobian norm at each step. From verify_density_results on pi0.5, `||J||_F` grows from ~11 to ~173 across ODE steps; even for this smaller Hopper MLP, the Jacobian at later steps (near t→1) may be large enough that `||J*dt||` is not small, making the first-order approximation poor. The next correction term `-tr(J^2)*dt^2/2` could be significant. Can use multiple Hutchinson samples to reduce variance but not bias.

**Compute cost**: O(D) per step per sample, vs O(D^3) for exact. Backward is clean (no matrix inverse).

---

### 3. Detach logdet, gradient through trajectory only

**Idea**: Compute exact KL value for logging, but `detach()` the logdet terms before adding to the loss. The gradient only flows through the trajectory (Path 1 + Path 3), not through slogdet (Path 2).

**Implementation**:
```python
sum_logdet_detached = sum_logdet.detach()
log_p_theta_approx = log_p0 - sum_logdet_detached  # exact value, no slogdet gradient
kl = log_p_theta_approx - log_p_base
# gradient from kl.mean() flows through log_p0 and log_p_base only
```

**Pros**: Exact KL value (for monitoring). Removes the dangerous Path 2 entirely. Gradient through log_p0 and trajectory is well-conditioned.

**Cons**: Gradient is almost trivial — `log_p0` depends on initial noise `z` (no parameter dependence) and `sum_logdet` is detached, so `log_p_theta` contributes **zero gradient** w.r.t. v_res parameters. The only KL gradient comes from `log_p_base` through the backward FP chain → `a_Km1` → ODE → v_res. This means the actor only learns how v_res affects the *base* policy's density evaluation at the combined policy's trajectory, not how it affects its own density. This makes the KL signal very weak and may be insufficient for meaningful regularization.

**Compute cost**: Same forward cost (still compute Jacobians for value), but backward is cheaper (no slogdet backward).

---

### 4. Per-step stop-gradient KL

**Idea**: Instead of accumulating logdet across all K-1 steps into one sum (which creates a deep computational graph), compute a local KL penalty at each step with the trajectory detached between steps.

**Implementation**:
```python
kl_total = 0
for k in range(K-1):
    a_k_detached = a_k.detach().requires_grad_(True)  # cut graph between steps
    J = jacrev(v_combined)(a_k_detached, t_k, cond)
    M = I + J * dt
    _, logdet = slogdet(M)
    kl_total += logdet  # gradient only through this step's Jacobian
    # advance with full graph for SAC path
    a_k = a_k + v_combined(a_k, t_k, cond) * dt
```

**Pros**: Limits the depth of the computational graph. Each step's slogdet backward only needs `(I + J_k * dt)^{-1}` at that step, not chained through all steps. May be more stable since conditioning of each step is independent.

**Cons**: Still uses slogdet backward (Path 2) — just locally. If `(I + J_k * dt)` is near-singular at any single step, it still blows up. Does not fundamentally eliminate the matrix inverse issue.

**Compute cost**: Same as exact, but potentially better-conditioned backward pass.

---

### 5. Spectral regularization on Jacobian

**Idea**: Prevent `(I + J*dt)` from becoming near-singular by explicitly penalizing the smallest singular value of `M = I + J*dt`. If `sigma_min(M) > epsilon`, the inverse is bounded by `1/epsilon`.

**Implementation**:
```python
M = I + J * dt
svd_vals = torch.linalg.svdvals(M)  # (B, D)
spectral_penalty = F.relu(epsilon - svd_vals.min(dim=-1).values).mean()
```

**Pros**: Directly addresses the root cause (near-singular M). Keeps the exact KL computation valid.

**Cons**: SVD is O(D^3) — same cost as slogdet. Adds another expensive operation. svdvals backward also requires the singular vectors, adding compute. This is a band-aid that makes the ill-conditioned path slightly less ill-conditioned, rather than eliminating it.

**Compute cost**: O(D^3) additional per step. Most expensive approach.

---

### 6. Velocity L2 penalty (replaces KL entirely)

**Idea**: Skip KL altogether. Penalize `||v_res||^2` directly. Since `v_res = 0` means `p_theta = p_base`, small v_res implies small KL. This is an upper bound on KL divergence rate (via Grönwall-type arguments for ODEs).

**Implementation**:
```python
# Already have v_res at each step from forward pass
v_res_penalty = 0
for k in range(K-1):
    v_k = self.v_res(a_k, t_k, cond)
    v_res_penalty += (v_k ** 2).sum(dim=(-2,-1)).mean()
v_res_penalty /= (K - 1)
total = sac_loss + vres_weight * v_res_penalty + sigma_ent_loss
```

**Pros**: No Jacobians at all. O(1) extra cost. Gradient is clean (just backprop through v_res forward pass). Simple.

**Cons**: Not KL — it's a proxy. Penalizes all directions equally, even beneficial ones. The relationship between `||v_res||^2` and KL depends on the base policy's Jacobian structure.

**Compute cost**: Negligible — just forward passes through v_res (already computed).

---

## Experiment plan

All experiments use: `alpha=0.1`, `sigma_entropy_weight=0.1`, `n_explore_steps=1000`, `critic_warmup_iters=5000`.

| Run | KL method | Key params | What we learn |
|-----|-----------|------------|---------------|
| A | None (baseline) | kl_weight=0, jac_weight=0 | Does alpha=0.1 alone prevent NaN? |
| B | Reward penalty (#1) | kl_reward_weight=0.05 | Does lagged KL signal work? |
| C | Hutchinson trace (#2) | kl_weight=0.05, hutchinson=True | Is trace approx stable + effective? |
| D | Detach logdet (#3) | kl_weight=0.05, detach_logdet=True | Does trajectory-only gradient suffice? |
| E | v_res L2 (#6) | vres_weight=0.01 | Simplest proxy — how close to real KL? |

Approaches #4 and #5 are deprioritized: #4 still has the matrix inverse issue; #5 adds cost without eliminating the root cause.

---

## Preliminary results (2026-05-03)

### Run A — Baseline (no KL, alpha=0.1): STABLE

- Survived past iter 17800+ with **zero NaN**. Confirms alpha=0.1 fixes the critic divergence.
- Reward climbed from ~1260 (base policy) to ~1660 by iter 17800, then some regression. Critic loss spiky (50–1200) but bounded.
- Previous no-KL run with alpha=0.0 died at iter 6921. **SAC entropy is the critical fix.**

### Run B — Reward penalty: in progress

- Still in critic warmup (iter ~5800) at time of writing. Actor not yet active.

### Run C — Hutchinson trace: NaN at iter 6973

- Critic went NaN first (iter 6973), ~973 actor iters after actor start at iter 6000.
- Actor was healthy at iter 6970 (loss -249, reward 1229), critic spiked to 288 → NaN.

### Run D — Detach logdet: NaN at iter 6329

- Critic went NaN first (iter 6329), ~329 actor iters after actor start at iter 6000.
- Actor was healthy at iter 6328 (loss -223, reward 1240), critic jumped to 55 → NaN.

### Run E — v_res L2: alive but degrading

- Alive at iter 9291, reward dropped to ~893 (from ~1260 base). Critic loss spiky (200–1400) but no NaN.

### Key finding: hidden slogdet in `backward_base_logprob` (Path 3)

Runs C and D died despite avoiding Path 2 (theta-side slogdet). Root cause: **`backward_base_logprob` also calls `slogdet`** on the base policy's Jacobian:

```python
# backward_base_logprob, lines 479-484:
J = vmap(jacrev(self._per_sample_base_velocity_flat, argnums=0))(a_k, ...)
M = I_D + J * dt
_, logabsdet = torch.linalg.slogdet(M)   # <-- base-side slogdet
```

The trajectory points `a_k` here are differentiable w.r.t. `a_Km1` (through the one-step refinement chain at lines 460-466). So backward through this slogdet still requires `(I + J_base * dt)^{-1}`. If the base policy's Jacobian makes this near-singular, the gradient explodes — **same mechanism as Path 2, just on the base policy side.**

This means the gradient anatomy has a **Path 3b** that was initially overlooked:

- **Path 3a** (benign): `kl_loss → log_p_base → log_p0(a^0_base) → refinement chain → a_Km1 → v_res`
- **Path 3b** (DANGEROUS): `kl_loss → log_p_base → slogdet(I + J_base * dt) → a_k → refinement chain → a_Km1 → v_res`

**All three KL-in-loss modes (exact, hutchinson, detach_logdet) share Path 3b.** The Hutchinson fix only addressed Path 2 (theta-side). The only approaches that avoid all dangerous gradient paths are:

| Approach | Path 2 (theta slogdet) | Path 3b (base slogdet) | Result |
|----------|----------------------|----------------------|--------|
| A: none | avoided | avoided | **stable** |
| B: reward_penalty | avoided (no_grad) | avoided (no_grad) | pending |
| C: hutchinson | avoided (JVP) | **still present** | NaN |
| D: detach_logdet | avoided (detach) | **still present** | NaN |
| E: vres_l2 | avoided | avoided | **stable** |

### Implication for fix

To make approaches 2–4 viable, `backward_base_logprob` must also avoid slogdet. Options:
1. Detach `a_k` in the base log-det loop (loses gradient through base density entirely)
2. Use Hutchinson trace for the base-side log-det too
3. Run the entire `backward_base_logprob` under `torch.no_grad()` (loses Path 3a too, but Path 3a may be too noisy to be useful anyway given the 30-iteration FP inversion)
