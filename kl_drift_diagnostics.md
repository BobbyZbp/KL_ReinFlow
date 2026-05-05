# KL Drift Diagnostics: Trajectory Divergence and Base Model Conditioning

## Hypothesis

As fine-tuning progresses, v_res grows, causing the combined policy's trajectory `a^k_theta` to drift away from where the base policy naturally maps noise. At some point, the combined trajectory enters regions where:
1. The base model's Jacobian becomes ill-conditioned → slogdet backward NaN
2. The base model assigns near-zero probability → KL explodes

## Experimental Setup

### Trajectory divergence & conditioning (script/diagnose_kl_drift.py)

For each checkpoint (at iters 0, 5k, 10k, 20k, 50k, 100k, 150k, 200k):
- Sample fixed noise `z ~ N(0, I)` (B=256, same across checkpoints)
- Run combined ODE: `a^{k+1}_theta = a^k + (v_base + v_res)(a^k, t_k) * dt`
- Run base ODE: `a^{k+1}_base = a^k + v_base(a^k, t_k) * dt`
- At each step k=0,1,2, measure:
  - Trajectory divergence: `||a^k_theta - a^k_base||`
  - `||v_res(a^k_theta, t_k)||` — residual magnitude
  - `cond(I + J_base(a^k_theta) * dt)` — base Jacobian conditioning at theta's trajectory
  - `cond(I + J_base(a^k_base) * dt)` — same at base's trajectory (control)
  - `sigma_min(I + J_base * dt)` — smallest singular value (proximity to singular)

**Note:** This script used observations sampled uniformly within the normalization range. This is acceptable for trajectory divergence and Jacobian conditioning measurements (which depend on the forward ODE, not on FP inversion), but the log_p_base values from this script are unreliable. See Section 3 for correct log_p_base analysis using real env observations.

### log_p_base decomposition with real observations (script/diagnose_logpbase_real.py)

For each checkpoint, collects 256 observations by rolling out that checkpoint's policy in Hopper-v2, normalizing via `normalization.npz` (min-max to [-1, 1]). Then:
- Runs forward combined ODE and base ODE from fixed noise
- At both endpoints (theta and base), attempts FP inversion with 30 iterations
- Decomposes: `log_p_base = log_p_0(a^0_recovered) - sum_k log|det(I + J_base * dt)|`
- Reports: neg_sign count (det < 0), round-trip error, a^0 norm, per-step logdet

## 1. Trajectory Divergence — Run A (baseline, no KL regularization)

### Trajectory divergence grows monotonically

```
    Iter | endpoint_div |  ||v_res|| k=2 |
---------+--------------+----------------+
        0 |       0.0000 |         0.0000 |
     5000 |       2.3277 |         6.5542 |
    10000 |       2.9047 |         8.6670 |
    20000 |       3.2318 |         9.0386 |
    50000 |       3.6048 |         9.6880 |
   100000 |       7.0602 |        18.4122 |
   150000 |      12.9709 |        35.2484 |
   199999 |      19.4028 |        54.4710 |
```

By 200k iters, the endpoint divergence is 19.4 (vs base endpoint norm of 11.4 — a 170% deviation). `||v_res||` grows to 54.5, roughly 3x the base velocity magnitude. The combined trajectory is in a completely different region of action space.

### Jacobian conditioning does NOT deteriorate at OOD points

This was the surprising finding. The base Jacobian at theta's trajectory is comparably conditioned to the base Jacobian at its own trajectory:

Column definitions — all measure properties of `M = I + J_base(a) · dt`, the matrix whose inverse in slogdet backward causes NaN:
- **cond@theta**: condition number of M with J_base evaluated at `a^k_theta` (combined policy's trajectory). Higher = closer to singular = more gradient amplification.
- **cond@base**: same, but J_base evaluated at `a^k_base` (base policy's own trajectory). Control: how conditioned is the Jacobian at points the base was trained on.
- **smin@theta**: smallest singular value of M at theta's trajectory. Directly measures proximity to singularity — if this hits 0, `M^{-1}` is infinite. At 0.04, the inverse amplifies gradients by ~25x.
- **smin@base**: same at base's own trajectory.

```
Step 2 (last deterministic step, k=2):

    Iter | cond@theta k=2 | cond@base k=2 | smin@theta k=2 | smin@base k=2
---------+----------------+---------------+----------------+--------------
        0 |         235.72 |        235.72 |     0.04151036 |   0.04151036
     5000 |         130.25 |        235.72 |     0.04191725 |   0.04151036
    50000 |          70.09 |        235.72 |     0.04726349 |   0.04151036
   100000 |         116.71 |        235.72 |     0.04578504 |   0.04151036
   199999 |         194.39 |        235.72 |     0.04743274 |   0.04151036
```

The condition number at theta's trajectory (70–194) is actually **lower** than at the base's own trajectory (236). The smallest singular value (~0.04–0.05) stays comparable. The Jacobian is poorly conditioned everywhere (sigma_min ~0.04 means the backward inverse amplifies by ~25x), but it's NOT meaningfully worse at OOD points.

### Worst-case conditioning is similar

```
    Iter | worst_cond@theta | worst_cond@base | worst_smin@theta | worst_smin@base
---------+------------------+-----------------+------------------+----------------
        0 |        15865.27  |       15865.27  |     0.00009015   |    0.00009015
     5000 |        11614.72  |       15865.27  |     0.00009828   |    0.00009015
   100000 |         3247.68  |       15865.27  |     0.00023435   |    0.00009015
   199999 |         7807.46  |       15865.27  |     0.00020035   |    0.00009015
```

The worst-case condition numbers at theta's trajectory are consistently lower than at the base's own trajectory. The base model's Jacobian is most ill-conditioned at its own trajectory — not at the drifted points.

## 1b. Trajectory Divergence — Run B (KL reward penalty)

Similar pattern. The KL reward penalty slows drift slightly (endpoint_div 14.1 at 190k vs 19.4 for Run A at 200k) but doesn't prevent it. The Jacobian conditioning shows the same pattern — no deterioration at OOD points.

```
    Iter | endpoint_div |  ||v_res|| k=2 | cond@theta k=2 | smin@theta k=2
---------+--------------+----------------+-----------------+---------------
        0 |       0.0000 |         0.0000 |          235.72 |    0.04151036
     5000 |       2.1581 |         5.9705 |          123.95 |    0.04206241
    50000 |       5.7704 |        15.8947 |          141.01 |    0.04697626
   100000 |       8.1775 |        23.6168 |          149.11 |    0.04563423
   190000 |      14.1401 |        45.6013 |          186.35 |    0.05298509
```

## 2. log_p_base Decomposition with Real Observations — Run A

Script: `script/diagnose_logpbase_real.py` — collects observations by rolling out each checkpoint's policy in Hopper-v2.

### The critical finding: det(I + J_base · dt) goes NEGATIVE

The determinant of the discrete Euler step matrix `M = I + J_base(a) · dt` is not just near-singular at the combined trajectory — it is **negative**. This means the discrete flow step reverses orientation: it is NOT a diffeomorphism. The change-of-variables formula `log p = log p_0 - sum log|det M|` requires `det M > 0` (orientation-preserving). When `det M < 0`, the FP inversion has no valid fixed point to converge to, and log_p_base is undefined.

- **neg_sign**: number of samples (out of 256) where `sign(det(I + J_base · dt)) < 0` at any step. This is the critical indicator — it means the discrete Euler step is not invertible at those points.

```
THETA endpoint (combined policy's action):

    Iter | valid/256 | neg_sign/256 | round-trip err | log_p_base mean
---------+-----------+--------------+----------------+----------------
        0 |   256/256 |        0/256 |       0.000005 |          -1.85
    20000 |    49/256 |      207/256 |            inf |           -inf
   100000 |     0/256 |      256/256 |            inf |           -inf
   199999 |     0/256 |      256/256 |            inf |           -inf

BASE endpoint (base policy's own action, control):

    Iter | valid/256 | neg_sign/256 | round-trip err | log_p_base mean
---------+-----------+--------------+----------------+----------------
        0 |   256/256 |        0/256 |       0.000005 |          -1.85
    20000 |   256/256 |        0/256 |       0.000017 |          -1.89
   100000 |   256/256 |        0/256 |       0.000038 |          -1.92
   199999 |   255/256 |        1/256 |       0.000055 |          -1.94
```

Key observations:
1. **iter 0** (v_res = 0, theta = base): Perfect. 0 neg_sign, round-trip error 5e-6, log_p_base ≈ -1.85.
2. **iter 20000**: 207/256 samples have negative determinant at theta's trajectory. FP inversion produces inf/NaN. Only 49 valid samples remain.
3. **iter 100000+**: 256/256 negative — EVERY sample. FP inversion completely fails. log_p_base is undefined.
4. **Base trajectory control**: 0/256 neg_sign through 100k iters; only 1/256 at 199k. The base flow's own trajectory stays invertible throughout.

### Per-step logdet at iter 0 (both trajectories identical)

```
    step 0 logdet: mean=-0.0181  min=-0.1113  max=0.0412  neg_sign=0/256
    step 1 logdet: mean=-0.2274  min=-0.7710  max=0.0247  neg_sign=0/256
    step 2 logdet: mean=-0.7858  min=-2.1789  max=-0.2139  neg_sign=0/256
```

All determinants positive, logdet contributions are modest (total sum_logdet ≈ -1.03), log_p_0 ≈ -0.82. The log_p_base ≈ -1.85 is a reasonable value.

### Per-step logdet at iter 20000 — THETA endpoint (49 valid samples only)

```
    step 0 logdet: mean=0.0091  min=-0.0685  max=0.0783  neg_sign=0/49
    step 1 logdet: mean=0.1170  min=-0.3072  max=0.6832  neg_sign=0/49
    step 2 logdet: mean=-0.3571  min=-2.3949  max=2.1035  neg_sign=0/49
```

Even among the 49 "valid" samples (where det stayed positive), the logdet variance is much higher (range [-2.4, 2.1] at step 2). The 207 invalid samples had det < 0 at one or more steps.

### Per-step logdet at iter 20000 — BASE endpoint (256 valid)

```
    step 0 logdet: mean=-0.0182  min=-0.1254  max=0.0418  neg_sign=0/256
    step 1 logdet: mean=-0.2288  min=-0.7805  max=0.0270  neg_sign=0/256
    step 2 logdet: mean=-0.8052  min=-2.3117  max=-0.2130  neg_sign=0/256
```

Nearly identical to iter 0. The base trajectory is stable.

## 3. Revised Understanding

### The real failure mode: orientation reversal, not ill-conditioning

The original hypothesis (H1: Jacobian becomes ill-conditioned at OOD points) is **not supported** — conditioning is similar everywhere.

The corrected diagnosis: at the combined policy's trajectory, the discrete Euler step `a^{k+1} = a^k + v_base(a^k, t_k) · dt` has `det(I + J_base · dt) < 0`. This means:

1. **The discrete map reverses orientation.** The continuous ODE is always a diffeomorphism (positive Jacobian determinant), but the Euler discretization with `dt = 0.25` is not. At the base policy's own trajectory, the Jacobian happens to have eigenvalues that keep `I + J · dt` positive-definite. At the drifted trajectory, some eigenvalues of `J_base · dt` are < -1, making `det(I + J · dt) < 0`.

2. **FP inversion has no solution.** The fixed-point iteration `a^k = a^{k+1} - dt · v_base(a^k, t_k)` seeks a fixed point of `g(a) = a^{k+1} - dt · v_base(a, t_k)`. When the discrete map reverses orientation, the Euler forward step from any `a^k` cannot reach `a^{k+1}` — the fixed point doesn't exist in the real number line.

3. **The change-of-variables formula is invalid.** `log p = log p_0(a^0) - Σ log|det(I + J · dt)|` assumes each step is a diffeomorphism (orientation-preserving). When det < 0, the formula produces `log|det|` which is finite but meaningless — the density it computes does not correspond to any well-defined probability.

### What the previous log_p_base ≈ -100,000 values actually were

The earlier analysis (using synthetic observations) claimed `log_p_base ≈ -100,000 nats` was "mathematically correct" and "inherent to flow-matching policies." **This was wrong.** Those values were numerical garbage from two compounding errors:

1. **Synthetic observations** are out-of-distribution for the trained model, causing FP to diverge at ALL checkpoints including iter 0.
2. **Negative determinants** at drifted trajectory points made the change-of-variables formula undefined, producing arbitrary large numbers.

With real observations, `log_p_base` at the base's own trajectory is a modest **-1.85 to -1.94 nats** across all checkpoints. The true KL (if it could be computed) is not 100,000 nats — but it genuinely cannot be computed via the discrete change-of-variables formula at the combined trajectory because the discrete flow is not invertible there.

### Whether better inversion methods would help

**No.** The problem is not numerical convergence of the fixed-point iteration. The problem is that the discrete Euler step is not a diffeomorphism at the combined trajectory. No inversion method (Newton, Anderson acceleration, adaptive ODE solvers) can find a pre-image that doesn't exist under the discrete map.

Two possible paths forward:
1. **More discretization steps** (smaller dt): With K=100 instead of K=4, `dt = 0.01`, and `I + J · 0.01` is much more likely to stay positive. But this makes training 25x more expensive and the Jacobian computation O(K · D³) per sample.
2. **Accept that exact discrete KL is intractable** for drifted trajectories, and use alternative proximity measures (see below).

### Implications for KL regularization

The KL between the fine-tuned and base policy **cannot be computed** via the discrete change-of-variables formula once training progresses past early iterations (~20k in our setup). This is not a numerical precision issue — the mathematical prerequisites (positive determinant) are violated.

The clamp at 50 (now raised to 1e4) was masking this: `backward_base_logprob` was returning garbage (inf, NaN, or arbitrary large negative numbers), and the clamp was silently truncating it. The wandb `kl_penalty_mean` plateau at 2.5 = clamp 50 × weight 0.05 was hiding a fundamentally broken computation.

### Better approaches

1. **Trajectory divergence constraint** (direct): Penalize `||a^k_theta - a^k_base||²` per step. Both trajectories start from the same noise, so this requires only one extra base-ODE forward pass (no inversion). The gradient is clean.

2. **Velocity divergence penalty**: Penalize `||v_res(a^k, t_k)||²` (what Run E does) or `||v_combined(a^k, t_k) - v_base(a^k, t_k)||²` evaluated at the combined trajectory. Same as trajectory constraint but local.

3. **Periodic rebasing**: Update the "base" policy to the current fine-tuned policy every N iterations. Keeps drift small between rebase steps.

4. **Reverse KL via score matching**: Approximate `∇_x log p_base(x)` (the base score function) using denoising score matching on base policy samples, without ever inverting the ODE.

5. **Wasserstein distance or MMD**: Compare distributional distance between base and theta samples directly, without density evaluation.

6. **Forward-only KL**: Drop `log_p_base` entirely (which is what we effectively did by detaching it). Accept that it's an entropy bonus, not KL. Combine with one of the above for a proper proximity constraint.

7. **More ODE steps**: Use K >> 4 (e.g., K=50–100) so that `dt` is small enough to keep `det(I + J · dt) > 0` everywhere. Makes the discrete flow a proper diffeomorphism but greatly increases compute cost.

## Raw Data

Full per-checkpoint results saved to:
- `log/kl_drift_diagnostics_runA.json`
- `log/kl_drift_diagnostics_runB.json`

Diagnostic scripts:
- `script/diagnose_kl_drift.py` — trajectory divergence and Jacobian conditioning
- `script/diagnose_logpbase_decomp.py` — log_p_base decomposition (uses synthetic obs, results unreliable)
- `script/diagnose_logpbase_real.py` — log_p_base decomposition with real env observations (correct)
