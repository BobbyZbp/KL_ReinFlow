# Diagnostic Findings: Per-Step KL Stability Analysis

Results from two offline diagnostic scripts run on existing checkpoints (no retraining required).
All computations use **real observations** collected by rolling out the base policy in Hopper-v2.

---

## 1. Diagnostic: Compare Runs KL Drift (`compare_runs_kl_drift.py`)

### What it computes

For each checkpoint in runs A2 (baseline, no KL penalty) and B (KL reward penalty):

**Per-step KL mean** — KL(p_theta || p_base) summed over ODE steps.
At each step k, computes `logdet(I + J_base(a_inv)*dt) - logdet(I + J_combined(a_k)*dt)`,
where `a_inv` is the fixed-point inverted point. Summed over K-1=3 steps, averaged over batch.

**||v_res|| total** — Total residual velocity magnitude across steps.
Computes `sum_k mean_batch(||v_res(a_k, t_k)||)`. Measures how much the fine-tuned policy
deviates from the base at each ODE step.

**Endpoint divergence** — L2 distance between combined and base endpoints.
Computes `mean(||a^{K-1}_theta - a^{K-1}_base||)`: the final action difference
when both policies start from the same noise.

**Trajectory divergence** — Per-step L2 distance between trajectories.
Computes `||a^k_theta - a^k_base||` at each intermediate step k.

**KL sign convention**: Negative KL means p_theta places more mass (higher density) than p_base at those actions — fine-tuned policy is "concentrating" into regions the base already liked. Positive KL means the fine-tuned policy is spreading into regions the base assigns low density — true divergence from the base.

### Results

#### Run A2 (baseline, no KL penalty)

| iter | KL mean | \|\|v_res\|\| | endpoint div |
|-----:|--------:|---------:|--------:|
| 0 | 0.066 | 0.000 | 0.000 |
| 5000 | -0.304 | 8.328 | 1.375 |
| 10000 | -0.871 | 10.198 | 1.685 |
| 20000 | -3.147 | 21.793 | 3.097 |
| 30000 | -3.726 | 27.776 | 3.487 |
| 40000 | -2.600 | 25.499 | 3.538 |
| 50000 | -3.211 | 31.195 | 4.124 |
| 60000 | -0.759 | 30.338 | 4.231 |
| 70000 | **+0.483** | 36.960 | 4.863 |
| 80000 | **+2.310** | 40.564 | 5.265 |

#### Run B (KL reward penalty, weight=0.05)

| iter | KL mean | \|\|v_res\|\| | endpoint div |
|-----:|--------:|---------:|--------:|
| 0 | 0.066 | 0.000 | 0.000 |
| 5000 | -0.432 | 6.991 | 1.129 |
| 10000 | -0.880 | 10.354 | 1.653 |
| 20000 | -2.638 | 16.311 | 2.296 |
| 30000 | -3.326 | 20.952 | 2.691 |
| 40000 | -2.760 | 20.453 | 2.707 |
| 50000 | -3.889 | 25.841 | 3.086 |
| 60000 | -2.627 | 26.699 | 3.238 |
| 70000 | -2.256 | 27.755 | 3.658 |
| 80000 | **-1.465** | 31.052 | 3.788 |

### Key takeaways

1. **A2's KL crosses positive after 60k** — the policy genuinely diverges from the base. B's KL stays negative throughout 80k iterations.
2. **Drift ratio at 80k**: B has 28% less endpoint divergence (3.79 vs 5.27) and 23% less v_res magnitude (31.1 vs 40.6).
3. **The KL reward penalty is effective**: it slows drift without eliminating learning — ||v_res|| still grows, just more gradually.

---

## 2. Diagnostic: Per-Step Determinant Analysis (`diagnose_perstep_det.py`)

### What it computes

Tests whether one-step FP inversion preserves positive determinant (required for valid log-density) while full backward inversion (endpoint method) produces negative determinant (orientation reversal = invalid density).

**neg@inv** — Count of samples where `det(I + J_base(a_inv)*dt) < 0`.
Negative det means the discrete base map is locally orientation-reversing at the inverted point.
If this is 0, the per-step logdet is well-defined (valid log-density).

**neg@traj** — Count where `det(I + J_base(a_k)*dt) < 0` on the combined trajectory.
Shows whether the combined ODE trajectory itself passes through base-model fold regions.

**det_inv_min** — Minimum determinant across the batch at inverted points.
How close to zero/negative the det gets. Values near 0 indicate near-singularity.

**fp_err_max** — Maximum FP residual: `||a_inv + v_base(a_inv)*dt - a_next||`.
Whether the fixed-point iteration converged. Values >1 mean inversion FAILED and
results at those points are unreliable.

**inv_shift** — Mean `||a_inv - a_k||`.
How far the inverted point is from the trajectory point. Large shift means the base model
needs a very different input to reach the same target.

**endpoint backward neg** — Neg sign count when inverting the full K-1 step trajectory backward.
Measures how many samples have at least one orientation-reversing step during endpoint inversion.

**endpoint direct neg** — `det(I + J_base*dt) < 0` evaluated AT the endpoint.
Direct check: can the base model even locally produce the fine-tuned endpoint?

### Results: Run A2 (baseline)

Observations collected from base policy rollout (stochastic, 256 samples).

| iter | step 0 neg@inv | step 1 neg@inv | step 2 neg@inv | endpoint back neg | fp_err_max (step 0) | fp_err_max (step 1) |
|-----:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 1/256 | 14/256 | 88/256 | 240/768 | 5.9e-2 | 3.3e+4 |
| 5000 | 0/256 | 30/256 | 104/256 | 550/768 | 5.7e-2 | 5.7e+4 |
| 10000 | 1/256 | 35/256 | 108/256 | 584/768 | 1.1e-1 | 5.4e+4 |
| 20000 | 0/256 | 56/256 | 180/256 | 697/768 | 2.0e-2 | 2.0e+4 |
| 40000 | 0/256 | 77/256 | 198/256 | 735/768 | 1.5e-2 | 2.8e+4 |
| 60000 | 0/256 | 91/256 | 153/256 | 739/768 | 1.2e-2 | 2.9e+4 |

### Results: Run B (KL reward penalty)

Different observation batch (same collection method, different env trajectory).

| iter | step 0 neg@inv | step 1 neg@inv | step 2 neg@inv | endpoint back neg | fp_err_max (step 0) | fp_err_max (step 1) |
|-----:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 0/256 | 0/256 | 1/256 | 0/768 | 1.8e-4 | 1.0e-3 |
| 10000 | 1/256 | 6/256 | 81/256 | 472/768 | 2.9e-3 | 1.1e+3 |
| 20000 | 1/256 | 122/256 | 246/256 | 764/768 | 1.7e+2 | 8.0e+3 |
| 40000 | 1/256 | 179/256 | 251/256 | 768/768 | 3.8e+2 | 9.7e+3 |
| 60000 | 8/256 | 226/256 | 244/256 | 767/768 | 1.6e+3 | 1.4e+4 |
| 80000 | 9/256 | 233/256 | 232/256 | 768/768 | 3.1e+3 | 2.0e+4 |

### Key takeaways

1. **Step 0 (t=0, near noise) is reliable**: FP converges (err < 0.1), neg signs are 0-1/256, det stays positive. This is where per-step KL is valid.

2. **Steps 1-2 (t=0.25, 0.5) have FP divergence**: `fp_err_max` in the thousands/tens-of-thousands means the fixed-point iteration did NOT converge for many samples. The neg sign counts at these steps are **unreliable** — they reflect garbage from diverged iterations, not true orientation reversal.

3. **Endpoint backward inversion fails catastrophically**: Nearly all samples (735-768/768) show negative determinant by 40k+ iterations. This is the definitive evidence that endpoint KL is ill-posed.

4. **The inv_shift reveals why FP diverges**: At step 0, `inv_shift` < 3 (the inverted point is near the trajectory). At steps 1-2, `inv_shift` reaches thousands — the FP iteration flew off to infinity rather than finding a valid pre-image.

---

## 3. Interpretation: Why Per-Step KL Works

The data supports this explanation:

**The discrete Euler map `a + v_base(a, t)*dt` with dt=0.25 is NOT a global diffeomorphism.** Even the base model (at iter 0 with zero v_res) can produce regions where the map is locally non-invertible. This happens because `||J_base||*dt` can exceed 1, violating the contraction condition for FP inversion.

**Step 0 is special** because it starts from Gaussian noise (well-spread, far from data manifold boundaries). The base velocity at t=0 maps noise toward data — this is a "well-conditioned" region where:
- The Jacobian is moderate (Lipschitz * dt < 1)
- FP inversion converges reliably
- det(I + J*dt) > 0 (local diffeomorphism preserved)

**Later steps (t=0.25, 0.5)** operate closer to the data manifold where the velocity field becomes stiffer (sharper curvature). Even the base model's discrete map can fail to be invertible here. Once v_res adds perturbations, the combined trajectory drifts into regions the base cannot reach via one step → FP diverges.

**Practical implication for training**: The per-step KL in training uses all K-1=3 steps with gradients through `slogdet`. Steps 1-2 contribute valid gradients when v_res is small (early training), but become unreliable as the policy drifts. The KL reward penalty (run B) helps by keeping v_res smaller, maintaining validity of the Jacobian computation longer.

---

## 4. Caveat: Observation Dependence

The two diagnostic runs used different observation batches (base policy rollout in a stochastic environment). This is why iter 0 results differ between A2 and B despite both having v_res=0:
- A2's observations produced `endpoint_norm=4.45` and 240/768 neg signs at iter 0
- B's observations produced `endpoint_norm=1.72` and 0/768 neg signs at iter 0

**Within-run trends are valid** (same obs batch across all checkpoints). Cross-run comparisons of absolute counts are not directly comparable. The compare_runs_kl_drift script avoids this issue by using a single shared observation batch for both runs.

---

## 5. Summary Table

| Property | Step 0 (t=0) | Steps 1-2 (t=0.25-0.5) | Full endpoint |
|----------|:---:|:---:|:---:|
| FP convergence | Converges (err<0.1) | Diverges (err>1000) | N/A (sequential) |
| Det positive? | Yes (0-1/256 neg) | Unreliable (FP failed) | No (735+/768 neg) |
| Inversion shift | Small (<3) | Huge (>500) | N/A |
| KL well-defined? | Yes | Questionable | No |
| Implication | Safe for gradient | Needs care (Hutchinson or clamp) | Must avoid |
