# KL Regularization: Three Orthogonal Design Axes

We want to minimize `KL(p_theta || p_base)` as a regularizer during SAC fine-tuning of a flow-matching policy. All experiments backpropagate through `log p_base` (no detaching). Three independent design choices control how KL is computed, how the logdet is estimated, and whether the base policy drifts.

## Setup recap

The combined policy runs a K-step ODE from noise `z ~ N(0, I)`:

```
a^{k+1} = a^k + (v_base(a^k, t_k) + v_res(a^k, t_k)) * dt
```

The true KL between endpoint marginals is:

```
KL = E_z[log p_theta(a^{K-1}_theta) - log p_base(a^{K-1}_theta)]
```

where `log p_base(a^{K-1}_theta)` requires **inverting the base ODE** to recover the base trajectory and initial noise. Its gradient decomposes into:

- **Term 1** (entropy): `d/d_theta log p_theta` -- pushes mass apart
- **Term 2** (cross-entropy): `-d/d_theta log p_base` -- pulls mass toward base modes

Both terms are kept live in all experiments below.

---

## Axis 1: KL scope (per-step vs endpoint)

**What it controls:** how `log p_base` is evaluated — full backward ODE inversion, or per-step single-step inversion.

### Endpoint KL (current implementation)

Compute log-densities at the final deterministic point `a^{K-1}`:

```
log p_theta = log p_0(z) - sum_k logdet(I + J_combined(a^k_theta) * dt)
log p_base  = log p_0(z_base) - sum_k logdet(I + J_base(a^k_base) * dt)
```

To compute `log p_base` at `a^{K-1}_theta`, we must **invert the base ODE**: find `z_base` such that `T_base(z_base) = a^{K-1}_theta`, recovering the full base trajectory `{a^k_base}_{k=0}^{K-1}`. This requires K-1 sequential fixed-point inversions, each needing ~10-20 FP iterations (see convergence data below). The Jacobians `J_base` are evaluated along the **recovered base trajectory**.

**Provides:** true `KL(p_theta || p_base)` (modulo logdet approximation). Both entropy and cross-entropy terms are exact.

**Requires:** full backward ODE inversion (K-1 sequential fixed-point inversions). Error accumulates across steps. Can diverge on out-of-distribution points as `v_res` grows.

### Per-step KL (one-step FP inversion per step)

Treat each ODE step as a **single-step MLP policy** and compute KL at that step independently. At step k, the combined transition is:

```
a^{k+1}_theta = a^k + (v_base(a^k, t_k) + v_res(a^k, t_k)) * dt
```

To compute the true single-step KL at step k, we need to find `a^k_inv` — the initial point that the base-only transition would map to the same output `a^{k+1}_theta`:

```
a^k_inv + v_base(a^k_inv, t_k) * dt = a^{k+1}_theta
```

This is solved by **one-step fixed-point iteration** (a single implicit Euler inversion, not full backward ODE):

```
a^k_inv_0     = a^{k+1}_theta - v_base(a^{k+1}_theta, t_k) * dt    (explicit estimate)
a^k_inv_{n+1} = a^{k+1}_theta - v_base(a^k_inv_n, t_k) * dt        (refine)
```

The per-step KL is then:

```
KL_k = logdet(I + J_base(a^k_inv) * dt) - logdet(I + J_combined(a^k) * dt)
       + log p^k(a^k) - log p^k(a^k_inv)
```

where `log p^k(a^k) - log p^k(a^k_inv)` is the density ratio of the input distribution at step k. This term is O(v_res * dt) and is **dropped** in practice — the two points differ by O(v_res * dt), and the density varies smoothly at that scale.

**Final per-step KL:**

```
KL_approx = sum_k [logdet(I + J_base(a^k_inv) * dt) - logdet(I + J_combined(a^k) * dt)]
```

**Key properties:**
- Only **one-step** FP inversion per step (not K-1 sequential inversions). No error accumulation across steps.
- J_base evaluated at the **correctly inverted point** `a^k_inv`, not at the combined trajectory `a^k`.
- Both logdet terms carry live gradients through `theta` (no detaching).
- The FP inversion itself runs under `torch.no_grad()` (or with one differentiable refinement step, as in the current `backward_base_logprob`).

### FP iteration count for one-step inversion

Data from pi0.5 (D=480, verify_density_results.md) shows per-step (1-step) reconstruction:

| FP iters | rel_err_real (1 step) |
|----------|----------------------|
| 5        | 1.76e-4              |

For the full 9-step backward inversion (error accumulates):

| FP iters | rel_err_real (9 steps) | logp_reldiff |
|----------|------------------------|-------------|
| 5        | 1.47%                  | 0.5-0.7%    |
| 10       | 1.0%                   | 0.05-0.11%  |
| 20       | 0.70%                  | 0.005-0.15% |

Per-step KL only does **one-step inversion**, so the 1.76e-4 error (5 FP iters) applies — no accumulation. For our D=12 problem (vs D=480), convergence should be even faster. **5-10 FP iterations per step should suffice** for per-step KL, compared to 10-20 needed for full backward inversion.

### Why one-step FP inversion is well-conditioned

The single-step map `a -> a + v_base(a, t) * dt` is a contraction when `||J_base * dt|| < 1` (Euler step with small dt). The FP iteration `a^k_inv_{n+1} = a^{k+1} - v_base(a^k_inv_n, t_k) * dt` converges geometrically with rate `||J_base * dt||_op`. For our setup:
- dt = 0.25, ||J_base||_F ~ 0.5-2.0 (D=12), so ||J_base * dt||_F ~ 0.1-0.5.
- Contraction is strong. Each FP iteration reduces error by ~2-10x.

### Why per-step is better motivated

For a single-step policy (an MLP) `a = z + v(z) * dt`, the true KL requires finding `z_inv` such that `z_inv + v_base(z_inv) * dt = z + v_combined(z) * dt`. This IS FP inversion — but only a single step. Per-step KL applies this single-step KL at each ODE step independently.

Compare to endpoint KL which chains K-1 inversions and accumulates error. Per-step KL decomposes the problem into K-1 independent, well-conditioned single-step inversions.

### Dropped density ratio term

The exact single-step KL at step k is:

```
KL_k = E_{a^k ~ p^k}[log p^k_theta(a^k) - log p^k_base(a^k)]
     = logdet(I + J_base(a^k_inv) * dt) - logdet(I + J_combined(a^k) * dt)
       + log p^k(a^k) - log p^k(a^k_inv)
```

The term `log p^k(a^k) - log p^k(a^k_inv)` accounts for the fact that the change-of-variables relates densities at different input points. Since `a^k_inv = a^k + O(v_res * dt)` and `log p^k` is smooth, this term is O(v_res * dt). We drop it, incurring a small bias that vanishes as v_res → 0 (start of training) and remains small under EMA (Axis 3).

---

## Axis 2: Logdet computation (exact slogdet vs Hutchinson trace)

**What it controls:** whether the backward pass involves matrix inverse (applies to BOTH `log p_theta` and `log p_base`).

When Hutchinson is used, it replaces slogdet on **both sides** -- for `J_combined` in `log p_theta` AND for `J_base` in `log p_base`. This eliminates slogdet backward (Path 2 and Path 3b) entirely.

### Exact slogdet

```python
J = vmap(jacrev(velocity_fn))(a, t, cond)
M = I + J * dt
_, logabsdet = torch.linalg.slogdet(M)     # O(D^3)
```

Backward pass: gradient of `slogdet(M)` w.r.t. `M` is `M^{-T}`. Explodes when `I + J * dt` is near-singular.

**Provides:** exact log-density values. Full Jacobian structure in gradient.

**Danger:** NaN from `M^{-1}` in backward pass. Applies to both the theta-side (Path 2) and base-side (Path 3b, for endpoint KL). For per-step KL, Path 3b is eliminated (no inversion), but `slogdet(I + J_base(a^k_theta) * dt)` backward can still NaN if `J_base` at combined trajectory points is poorly conditioned.

### Hutchinson trace estimator

```python
v_probe = rademacher(B, D)
Jv = jvp(velocity_fn, (a,), (v_probe,))[1]   # O(D), GPU-parallelizable
tr_est = (v_probe * Jv).sum(dim=-1)
logdet_approx = tr_est * dt                   # log|det(I+M)| ~ tr(M) for small ||M||
```

Replaces `log|det(I + J*dt)|` with `tr(J) * dt` (first-order Taylor: `log det(I+M) ~ tr(M)`). Applied to BOTH `J_combined` and `J_base`.

Backward pass: gradient through a JVP is a VJP composition. No matrix inverse. Fully NaN-safe.

**Provides:** NaN safety on both sides (Path 2 and Path 3b eliminated). O(D) cost per probe, GPU-parallelizable across probes.

**Loses:** exactness. The first-order Taylor is biased when `||J*dt||` is not small. Hutchinson adds variance (reduced by more probes, but not the Taylor bias).

### Important: Hutchinson does NOT cancel with per-step KL (one-step inversion)

With the one-step FP inversion formulation, J_base is evaluated at `a^k_inv` while J_combined is evaluated at `a^k`. Since `a^k_inv ≠ a^k`, the Hutchinson traces do NOT cancel:

```
KL_per_step_hutch = sum_k [tr(J_base(a^k_inv)) - tr(J_combined(a^k))] * dt
                  = sum_k [tr(J_base(a^k_inv)) - tr(J_base(a^k)) - tr(J_res(a^k))] * dt
```

The first two terms differ because `a^k_inv = a^k + O(v_res * dt)`. This difference carries cross-entropy signal:

```
tr(J_base(a^k_inv)) - tr(J_base(a^k)) ≈ [d tr(J_base)/da] · (a^k_inv - a^k) = O(v_res * dt)
```

So even with Hutchinson, the per-step KL retains a cross-entropy signal proportional to v_res — unlike the naive formulation (both Jacobians at same point) which would collapse to `-sum tr(J_res) * dt`.

**Contrast with naive per-step (no inversion):** If both Jacobians were evaluated at `a^k` (no FP inversion), then `J_base(a^k)` cancels and the KL reduces to `-sum tr(J_res) * dt` — a pure divergence penalty with no cross-entropy signal from the base policy. The one-step FP inversion is what preserves the base-policy signal.

With exact slogdet, the nonlinear logdet structure provides even richer signal, but the key point is that **Hutchinson + one-step inversion still works** — it does not degenerate.

---

## Axis 3: Base policy (fixed vs periodic EMA)

**What it controls:** how far the combined trajectory can drift from the base trajectory over training.

### Fixed base

`v_base` is loaded once (pretrained weights) and frozen. As `v_res` grows:
- Combined trajectory diverges from what `v_base` was trained on.
- `J_base` at combined trajectory points becomes unpredictable.
- Fixed-point inversion (endpoint KL) converges slowly or diverges.
- Per-step approximation quality degrades.

### Periodic EMA (absorb residual into base)

```python
# every N actor updates:
with torch.no_grad():
    for p_base, p_res in zip(v_base.parameters(), v_res.parameters()):
        p_base.data += tau * p_res.data
        p_res.data *= (1 - tau)
```

Parameter-space interpolation. Not exact for nonlinear MLPs (`v_base(x; theta_b + tau*theta_r) != v_base(x; theta_b) + tau*v_res(x; theta_r)`), but effective when `v_res` is small.

**Provides:** bounded `v_res` magnitude, keeping per-step approximation valid and backward inversion stable. Trust-region-style constraint (TRPO/PPO analogy).

**Loses:** fixed reference. KL is measured against a moving target.

**Limitation:** requires `v_base` and `v_res` to have identical architectures (same param shapes). In our setup, `v_base` is [512,512,512] (553K params) and `v_res` is [128,128] (24K params) — absorption is impossible.

### LoRA-like reference policy (EMA of v_res)

When `v_base` and `v_res` have different architectures, we cannot absorb `v_res` into `v_base`. Instead, we maintain a **reference residual** `ref_v_res` with the same architecture as `v_res`:

```
reference policy:  v_ref(a, t, s) = v_base(a, t, s) + ref_v_res(a, t, s)
combined policy:   v_theta(a, t, s) = v_base(a, t, s) + v_res(a, t, s)
```

`ref_v_res` is initialized as a copy of `v_res` (zero-init at start), kept frozen (`requires_grad=False`, eval mode), and periodically updated via EMA:

```python
# every N actor updates:
with torch.no_grad():
    for p_ref, p_res in zip(ref_v_res.parameters(), v_res.parameters()):
        p_ref.data.mul_(1 - tau).add_(p_res.data, alpha=tau)
```

The KL is then computed as `KL(p_theta || p_ref)` instead of `KL(p_theta || p_base)`. At each per-step KL computation, FP inversion uses `v_ref` instead of `v_base`:

```
a^k_inv + v_ref(a^k_inv, t_k) * dt = a^{k+1}_theta
KL_k = logdet(I + J_ref(a^k_inv) * dt) - logdet(I + J_theta(a^k) * dt)
```

**Key properties:**
- Same architecture for `ref_v_res` and `v_res` — EMA is exact (same param shapes).
- At init, `ref_v_res = v_res = 0`, so `v_ref = v_base` and `KL = 0` (correct).
- As training progresses, `ref_v_res` tracks `v_res` with a lag, so `v_ref ≈ v_theta` and KL stays small — the trust region moves with the policy.
- FP inversion uses `v_ref` (which includes `v_base`), so it remains well-conditioned as long as `ref_v_res - v_res` is small.
- `v_base` stays frozen and untouched — no architecture mismatch issues.

**Analogy to LoRA:** in LoRA fine-tuning, the base model is frozen and a low-rank residual is trained. The reference for KL in RLHF is the LoRA model at init (or an EMA). Here, `v_base` is the frozen base, `v_res` is the "LoRA adapter," and `ref_v_res` is the reference adapter.

**kl_mode values:**
- `perstep_exact_ref` — per-step KL vs reference policy, exact slogdet
- `perstep_hutchinson_ref` — per-step KL vs reference policy, Hutchinson trace

**Config:**
```yaml
train:
  kl_mode: perstep_exact_ref
  ema_absorb_freq: 100    # update ref_v_res every 100 actor updates
  ema_absorb_tau: 0.01    # EMA rate
```

---

## The 8 combinations

| # | KL scope | logdet | base | Key property |
|---|----------|--------|------|--------------|
| 1 | endpoint | exact | fixed | True KL, full danger. K-1 sequential FP inversions + slogdet backward. NaN from Path 2 + 3b + inversion divergence. |
| 2 | endpoint | Hutchinson | fixed | True KL (approx logdet). No slogdet NaN. K-1 inversions can still diverge. |
| 3 | endpoint | exact | EMA | True KL. EMA tames inversion. slogdet NaN still possible. |
| 4 | endpoint | Hutchinson | EMA | True KL (approx logdet). Safest endpoint variant: no slogdet NaN, EMA stabilizes inversion. |
| 5 | per-step | exact | fixed | Per-step KL with one-step FP inversion. slogdet backward possible but well-conditioned (single step). Cross-entropy signal retained via a^k_inv. |
| 6 | per-step | Hutchinson | fixed | Per-step KL with one-step FP inversion + Hutchinson. No slogdet backward. Cross-entropy signal retained (a^k_inv ≠ a^k prevents cancellation). Fully NaN-safe. |
| 7 | per-step | exact | EMA ref | Like run 5 + EMA reference policy (LoRA-style). KL vs `v_base + ref_v_res` with EMA-updated `ref_v_res`. Trust region moves with policy. `kl_mode=perstep_exact_ref`. |
| 8 | per-step | Hutchinson | EMA ref | Like run 6 + EMA reference policy. Fully NaN-safe + drift control. Most conservative. `kl_mode=perstep_hutchinson_ref`. |

### Safety classification

**Fully NaN-safe:** 6, 8 (per-step + Hutchinson). No slogdet backward, one-step FP inversion is well-conditioned (contraction rate `||J_base * dt||`).

**Mostly safe:** 5, 7 (per-step + exact). One-step FP inversion well-conditioned. slogdet backward through single-step `logdet(I + J*dt)` — safer than endpoint (single step, not accumulated). EMA (run 7) adds a safety net.

**Partially safe:** 2, 4 (endpoint + Hutchinson). No slogdet NaN, but K-1 sequential FP inversions can diverge (error accumulates). EMA (run 4) mitigates.

**Dangerous:** 1, 3 (endpoint + exact). Full NaN exposure from slogdet backward + K-1 inversions. EMA (run 3) helps but doesn't eliminate.

### Experimental results (2026-05-04)

**Endpoint exact slogdet is fatal.** Both exact-slogdet endpoint runs NaN'd early:

| Run | Combo | NaN at iter | Reward at death | Cause |
|-----|-------|-------------|-----------------|-------|
| 11 | endpoint + exact + fixed | ~6,700 | 148 | slogdet backward → Q target NaN |
| 13 | endpoint + exact + EMA ref | ~8,900 | 148 | same; EMA bought ~2k more iters |

**Endpoint Hutchinson is stable.** Both Hutchinson endpoint runs survived:

| Run | Combo | Status at iter 9k | Reward |
|-----|-------|--------------------|--------|
| 12 | endpoint + Hutchinson + fixed | healthy | 891 |
| 14 | endpoint + Hutchinson + EMA ref | healthy | 873 |

**Per-step exact slogdet is stable.** Runs 5, 5b, 7, 8 (per-step + exact + fixed, various kl_weights) all ran to 30-45k+ iterations without NaN. The single-step `logdet(I + J*dt)` is far better conditioned than the accumulated endpoint version.

**Key finding:** the NaN killer is `slogdet` backward (requiring `M^{-1}` where `M = I + J*dt`), NOT the FP inversion or the endpoint formulation itself. Hutchinson eliminates slogdet backward entirely. Per-step exact also avoids it because single-step `I + J*dt` stays well-conditioned (eigenvalues bounded away from 0 for small `||J*dt||`), while endpoint accumulates K-1 steps of Jacobian products that can become near-singular.

### Why slogdet backward is the NaN killer

The **forward** computation `slogdet(M)` returns `log|det(M)|`, which is finite as long as `det(M) ≠ 0`. The **backward** is the problem. The gradient of `log|det(M)|` w.r.t. `M` is:

```
∂/∂M log|det(M)| = M^{-T}
```

This requires **inverting** `M = I + J·dt`. If any eigenvalue of `J·dt` is near `-1`, then `M` has a near-zero eigenvalue, `M^{-1}` explodes → NaN gradients → NaN actor loss → NaN poisons Q targets → permanent death.

**Why endpoint is worse than per-step:** For endpoint KL, the Jacobian `J` is evaluated at points along the combined trajectory, which drifts from the base policy's training distribution as `v_res` grows. Out-of-distribution points can produce `J` with large negative eigenvalues. The endpoint formulation compounds this across K-1 steps — one bad Jacobian anywhere in the chain kills the entire gradient.

For per-step KL, each `logdet(I + J·dt)` is independent. With `dt = 0.25` and `||J||` moderate at early training, the eigenvalues of `I + J·dt` stay bounded away from 0. That's why per-step exact ran 40k+ iterations without NaN while endpoint exact died at ~7k.

### How Hutchinson avoids slogdet backward

Hutchinson replaces `log|det(I + J·dt)|` with a first-order Taylor approximation:

```
log det(I + M) ≈ tr(M) = tr(J)·dt
```

The trace is estimated via `v^T J v` using JVPs (Jacobian-vector products):

```python
v_probe = rademacher(B, D)
_, Jv = torch.func.jvp(vel_fn, (a,), (v_probe,))
tr_est = (v_probe * Jv).sum(dim=-1)
```

The backward of a JVP is a VJP composition — **standard chain rule, no matrix inverse anywhere.** No `M^{-1}`, no slogdet, no near-singular matrices. Fully NaN-safe by construction.

### Hutchinson Taylor error for discrete change of variables

The Taylor approximation `log det(I + M) ≈ tr(M)` IS biased. The exact expansion is:

```
log det(I + M) = tr(M) - tr(M²)/2 + tr(M³)/3 - ...
```

Hutchinson only uses the first term `tr(M)`. The error is `O(||M||²)` where `M = J·dt`. For our setup (`dt = 0.25`, `||J||_F ≈ 2-6` depending on training stage):

| Training stage | `||J·dt||_F` | Taylor rel. error |
|---|---|---|
| Early (iter ~1k) | ~0.5 | 3-4% |
| Mid (iter ~100k) | ~1.0 | 8-10% |
| Late (iter ~200k) | ~1.5 | 15-30% |

**For KL regularization, this bias is mostly harmless.** The KL weight `λ` is a tuned hyperparameter — a 20% biased KL at `λ=0.05` is functionally equivalent to exact KL at `λ=0.04`. The gradient still pushes in the right direction (penalizing divergence from base), even if the magnitude is off. The weight is tuned empirically regardless.

**Could Hutchinson be made exact?** Yes — use higher-order trace terms:

```
tr(M²) via v^T M² v = v^T M (M v)    — 2 sequential JVPs
tr(M³) via v^T M³ v                   — 3 sequential JVPs
```

But for D=12, exact `jacrev` costs 12 backward passes and gives the **full Jacobian** + **exact logdet**. Higher-order Hutchinson with even 2 terms costs `2 × n_probes` JVPs. The exact Jacobian is cheaper and more accurate at this problem size. Hutchinson's advantage only kicks in at large D (like D=480 in pi0.5, where 32-64 JVPs << 480 backward passes).

**Bottom line for D=12 Hopper:** use per-step exact (combo 5) — it's stable, accurate, and cheaper than multi-probe Hutchinson. Hutchinson is the fallback for when exact slogdet NaNs (endpoint KL) or for scaling to larger action spaces.

### Which to prioritize

**Run 5** (per-step + exact + fixed): the most informative experiment. One-step FP inversion is well-conditioned, slogdet on a single step is safer than endpoint. Retains full nonlinear logdet structure and true cross-entropy signal. Tests whether per-step KL with exact logdet is stable in practice.

**Run 6** (per-step + Hutchinson + fixed): fully NaN-safe fallback. If run 5 NaNs, this confirms whether the issue is slogdet backward or the FP inversion. With one-step inversion, Hutchinson traces don't cancel — cross-entropy signal survives.

**Run 7** (per-step + exact + EMA): if run 5 is stable but KL grows as training progresses, EMA keeps v_res bounded so the per-step approximation stays accurate.

**Run 4** (endpoint + Hutchinson + EMA): richest signal (true KL). Tests whether EMA keeps K-1 inversions from diverging. Compare against per-step runs to measure the approximation cost.

---

## Policy gradient with Hutchinson-estimated log p

**Question:** can we use policy gradient (REINFORCE) instead of reparameterization to get KL gradients from Hutchinson-estimated probabilities?

### The idea

The reparameterization gradient of the cross-entropy term requires `nabla_x log p_base(x)` (the score of the base policy), which involves slogdet backward or backward inversion. REINFORCE avoids this entirely:

```
Reparameterization:  nabla_theta E_{p_theta}[-log p_base(x)]
                   = E_z[-nabla_x log p_base(x) * nabla_theta T_theta(z)]
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     requires score of p_base (dangerous)

REINFORCE:           nabla_theta E_{p_theta}[-log p_base(x)]
                   = E_{p_theta}[-log p_base(x) * nabla_theta log p_theta(x)]
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^
                                                   requires score of p_theta (safe with Hutchinson)
```

REINFORCE needs:
1. `nabla_theta log p_theta(x)`: the score function of the trained policy. With Hutchinson, this is computed by differentiating through the JVP trace estimator -- no slogdet, no matrix inverse.
2. `log p_base(x)`: the VALUE of log p_base (scalar, no gradient). Can be computed under `torch.no_grad()` via backward inversion + exact slogdet. No NaN risk since no backward pass.

This gives the cross-entropy gradient without ANY gradient through backward inversion or slogdet.

### How it works in practice

```python
# 1. Sample action via reparameterization (for SAC Q-value term)
z = torch.randn(B, ...)
a_K, a_Km1, eps = forward_ode(z, cond)  # reparameterized, attached to graph

# 2. SAC loss (reparameterization, standard)
sac_loss = -Q(s, a_K)

# 3. Compute log p_theta with Hutchinson (gradient through v_res)
log_p_theta_hutch = log_p0(z) - sum_k hutchinson_trace(J_combined, a^k) * dt

# 4. Compute log p_base under no_grad (just the value)
with torch.no_grad():
    log_p_base = backward_base_logprob(a_Km1, cond)  # exact, safe

# 5. KL via REINFORCE: score of p_theta times KL value
kl_value = (log_p_theta_hutch - log_p_base).detach()  # stop gradient on KL value
reinforce_kl = kl_value * log_p_theta_hutch            # REINFORCE estimator

# 6. Total actor loss
total = sac_loss + kl_weight * reinforce_kl.mean() + entropy + sigma_ent
```

### Tradeoffs

**Advantages:**
- No gradient through backward inversion (safe)
- No slogdet backward (safe)
- Uses the TRUE KL value (from exact computation under no_grad) for the REINFORCE weight
- Compatible with reparameterization for the SAC Q-value term

**Disadvantages:**
- REINFORCE has much higher variance than reparameterization
- The Hutchinson-estimated score `nabla_theta log p_theta` is noisy (Hutchinson variance + Taylor bias)
- The `kl_value * log_p_theta` product amplifies variance (product of two noisy quantities)
- Requires careful baseline/variance reduction to be practical

### Verdict

REINFORCE for the KL term is theoretically clean and avoids all NaN paths. But the variance penalty is severe -- especially combined with Hutchinson noise in the score function. It could work with:
- Many Hutchinson probes (reduce score variance)
- A learned baseline `b(s)` to reduce REINFORCE variance
- Only applying REINFORCE to the cross-entropy term (use reparameterization for entropy)

This is a viable 4th axis but adds significant complexity. Recommend testing per-step + exact (run 5) and per-step + Hutchinson (run 6) first, since they are simpler and may already provide sufficient signal.

---

## Empirical: Hutchinson probe count vs accuracy (real checkpoints)

Script: `script/test_hutchinson_accuracy.py`. Run A checkpoints (iters 0, 100k, 200k), B=64, 100 trials (6400 samples per step), real D4RL observations.

### Two independent error sources

The Hutchinson logdet approximation `log|det(I + J·dt)| ≈ tr_hutch(J) · dt` introduces two distinct errors:

**1. Taylor approximation error** — replacing `log det(I + J·dt)` with `tr(J)·dt`. This is a first-order Taylor expansion of the matrix logarithm:

```
log det(I + M) = tr(M) - tr(M²)/2 + tr(M³)/3 - ...
```

The zeroth-order term `tr(M)` is exact only when `||M|| → 0`. The bias is `O(||J·dt||²)` and does NOT decrease with more Hutchinson probes. It is a systematic, irreducible error inherent to the approximation.

**2. Hutchinson sampling error** — estimating `tr(J)` via `v^T J v` with Rademacher probes. This is an unbiased estimator with variance `2·||J||_F² / n_probes`. The RMSE decreases as `O(1/sqrt(n_probes))`. With enough probes, this error vanishes entirely.

The total logdet error is the sum of both. Increasing probes only reduces error source (2). If the Taylor bias is large, more probes yield diminishing returns — the error plateaus at the Taylor floor.

### Results: J_res (the one that matters for Hutchinson KL)

Since `KL_hutch = -sum_k tr(J_res(a^k)) · dt` (base Jacobian terms cancel by linearity of trace), the accuracy of Hutchinson KL depends entirely on estimating `tr(J_res)`.

**Taylor approximation quality for J_res:**

| Checkpoint | Step | `||J_res||_F` | `||J_res·dt||_F` | Taylor |rel err| | Taylor bias |
|------------|------|--------------|-------------------|-----------------|-------------|
| iter 0 | all | 0.000 | 0.000 | 0% (zero-init) | 0 |
| iter 100k | k=0 | 2.99 | 0.75 | 2.7% | 0.003 |
| iter 100k | k=1 | 3.88 | 0.97 | 3.3% | -0.015 |
| iter 100k | k=2 | 3.94 | 0.98 | 4.4% | -0.007 |
| iter 200k | k=0 | 4.88 | 1.22 | 28.5% | 0.526 |
| iter 200k | k=1 | 6.56 | 1.64 | 7.7% | 0.096 |
| iter 200k | k=2 | 6.28 | 1.57 | 9.4% | 0.079 |

At iter 100k, `||J_res·dt||_F ≈ 0.75–0.98` — the Taylor expansion is reasonable (3–4% error). By iter 200k, `||J_res·dt||_F ≈ 1.2–1.6` — the Taylor bias grows to 8–29%. The first-order approximation breaks down as `v_res` grows.

**Hutchinson probe count vs logdet relative error for J_res (iter 100k, step k=1):**

```
n_probes | trace_rmse | trace_rel% | logdet_rmse | logdet_rel%
---------|------------|------------|-------------|------------
       1 |     3.2335 |     128.1% |    0.809020 |      131.4%
       2 |     2.2753 |      90.1% |    0.569388 |       92.5%
       4 |     1.6030 |      63.5% |    0.402029 |       65.3%
       8 |     1.1489 |      45.5% |    0.288629 |       46.9%
      16 |     0.8100 |      32.1% |    0.204499 |       33.2%
      32 |     0.5758 |      22.8% |    0.146215 |       23.7%
      64 |     0.4068 |      16.1% |    0.104402 |       17.0%
     128 |     0.2878 |      11.4% |    0.076178 |       12.4%
     256 |     0.2053 |       8.1% |    0.057596 |        9.4%
     512 |     0.1446 |       5.7% |    0.044242 |        7.2%
```

The trace RMSE decays as ~1/sqrt(n_probes) as expected. With 1 probe, the Hutchinson error alone is 128% — the estimate is essentially noise. You need ~128 probes to get below 15% relative error.

**Why J_res is hard to estimate:** `||J_res||_F ≈ 3–6` but `|tr(J_res)| ≈ 2–5`. The Frobenius norm (which determines Hutchinson variance) is comparable to the trace magnitude. Hutchinson variance is `2·||J||_F²/n_probes`, so variance ∝ `||J||_F²` while signal is `|tr(J)|²`. The SNR scales as `|tr(J)|²·n_probes / (2·||J||_F²)`. For our J_res, `|tr/||J||_F| ≈ 0.7–0.8`, so SNR ≈ `0.25·n_probes`. You need `n_probes ≈ 4/SNR ≈ 16` just to get SNR > 1, and ~128 for 10% accuracy.

### Results: J_combined and J_base (for reference)

For J_base and J_combined, the Taylor error dominates completely. Even with 512 probes, logdet_rel% barely drops below the Taylor floor:

**J_combined at iter 200k:**

| Step | `||J·dt||_F` | Taylor |rel err| | 1-probe logdet_rel% | 512-probe logdet_rel% |
|------|-------------|-----------------|---------------------|----------------------|
| k=0 | 1.73 | 38.4% | 46.3% | 43.1% |
| k=1 | 2.54 | 29.5% | 37.2% | 33.3% |
| k=2 | 3.79 | 15.3% | 24.4% | 20.0% |

The 512-probe numbers barely improve over 32-probe — the error is pinned at the Taylor floor. `||J·dt||_F = 1.7–3.8` is far too large for the first-order approximation. The correction terms `-tr(M²)/2 + tr(M³)/3 - ...` are non-negligible.

### Conclusions

1. **For J_res (what Hutchinson KL actually uses):** Taylor bias is 3–29% depending on training stage. Hutchinson sampling error requires ~128 probes to reach 12% — but each probe costs one JVP (equivalent to one backward pass). With D=12, exact `jacrev` costs 12 backward passes and gives exact trace AND exact logdet. **Exact Jacobian is cheaper and more accurate** for this problem size.

2. **For J_combined/J_base:** Taylor bias is 15–38%, making Hutchinson fundamentally unable to estimate logdet accurately regardless of probe count. These Jacobians appear in the per-step KL formulation. If using per-step KL, exact slogdet is necessary for accurate density computation (the Hutchinson approximation is too biased).

3. **Scaling to larger D:** The break-even point where Hutchinson becomes cheaper than exact Jacobian is roughly `n_probes < D`. For D=12, that's 12 probes — which gives ~40–60% relative error. For larger models (D=480 in pi0.5), Hutchinson with 32–64 probes would be both cheaper (64 JVPs vs 480 backward passes) and reasonably accurate, since `||J_res||_F` likely scales sublinearly with D while `|tr(J_res)|` scales linearly. **The Hutchinson approach may be viable for pi0.5 but is a poor fit for D=12 Hopper.**
