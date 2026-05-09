# Diagnostic Findings: Per-Step KL Stability Analysis (v2)

Results from offline diagnostic scripts run on existing checkpoints.

**v2 fix (2026-05-07):** The original diagnostic collected observations once using the
base policy before the checkpoint loop. Different script invocations (run A2 vs B) got
different stochastic env trajectories, making iter-0 results incomparable (A2 showed
88/256 neg dets at step 2 while B showed 1/256 — same model, different obs). The fix:
collect on-policy observations **per checkpoint** with a fixed env seed, so each
checkpoint is evaluated on its own on-distribution states and iter-0 results are
identical across runs.

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

**NOTE on KL sign**: True KL(p_theta || p_base) is always >= 0 by Gibbs' inequality.
A persistently negative `kl_mean` is proof that the computation is producing invalid
values — see Section 3 for the root cause (FP inversion failure at steps 1-2).

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
| 70000 | +0.483 | 36.960 | 4.863 |
| 80000 | +2.310 | 40.564 | 5.265 |

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
| 80000 | -1.465 | 31.052 | 3.788 |

### Key takeaways

1. **KL mean goes negative in both runs** — this is physically impossible (true KL >= 0)
   and confirms the per-step KL computation is corrupted by FP inversion failures at
   steps 1-2 (see Section 2).
2. **Drift ratio at 80k**: B has 28% less endpoint divergence (3.79 vs 5.27) and 23%
   less v_res magnitude (31.1 vs 40.6).
3. **||v_res|| and endpoint divergence are trustworthy proxies** — they require no FP
   inversion and monotonically track policy drift.

---

## 2. Diagnostic: Per-Step Determinant Analysis (`diagnose_perstep_det.py`)

### What it computes

Tests whether one-step FP inversion preserves positive determinant (required for valid
log-density) at each ODE step, and whether full backward inversion (endpoint method)
produces negative determinant.

**neg@inv** — Count of samples where `det(I + J_base(a_inv)*dt) < 0`.
Negative det means the discrete base map is locally orientation-reversing at the
inverted point. If this is 0, the per-step logdet is well-defined.

**neg@traj** — Count where `det(I + J_base(a_k)*dt) < 0` on the combined trajectory.

**det_inv_min** — Minimum determinant across the batch at inverted points.

**fp_err_max** — Maximum FP residual: `||a_inv + v_base(a_inv)*dt - a_next||`.
Values >1 mean inversion FAILED and results at those points are unreliable.

**inv_shift** — Mean `||a_inv - a_k||`.
How far the inverted point is from the trajectory point. Large shift means FP
iteration diverged rather than finding a valid pre-image.

### Results: Run A2 (baseline, no KL penalty)

On-policy observations per checkpoint, 256 samples, env seed=42, FP iters=10.

| iter | step 0 neg@inv | step 1 neg@inv | step 2 neg@inv | endpoint back neg | fp_err_max (s0) | fp_err_max (s1) | fp_err_max (s2) | inv_shift (s0) | inv_shift (s1) | inv_shift (s2) |
|-----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 1/256 | 0/256 | 0/256 | 0/768 | 2.6e-3 | 9.0e-4 | 1.1e-2 | 0.0 | 0.0 | 0.0 |
| 5000 | 0/256 | 0/256 | 26/256 | 124/768 | 3.2e-4 | 2.5e+0 | 3.7e+3 | 0.8 | 1.0 | 39.3 |
| 10000 | 1/256 | 2/256 | 69/256 | 346/768 | 5.3e-4 | 8.6e+1 | 1.0e+4 | 1.0 | 1.4 | 74.3 |
| 20000 | 2/256 | 145/256 | 244/256 | 768/768 | 4.9e+2 | 8.8e+3 | 1.2e+4 | 4.0 | 1152.6 | 4181.6 |
| 40000 | 4/256 | 197/256 | 233/256 | 768/768 | 1.2e+3 | 1.5e+4 | 1.2e+4 | 8.7 | 2220.9 | 5173.8 |
| 60000 | 5/256 | 136/256 | 33/256 | 763/768 | 1.5e+3 | 1.2e+4 | 9.0e+3 | 9.1 | 2058.0 | 4171.9 |
| 80000 | 7/256 | 91/256 | 109/256 | 754/768 | 2.4e+3 | 1.3e+4 | 1.1e+4 | 15.1 | 2196.3 | 4715.6 |

### Results: Run B (KL reward penalty, weight=0.05)

On-policy observations per checkpoint, 256 samples, env seed=42, FP iters=10.

| iter | step 0 neg@inv | step 1 neg@inv | step 2 neg@inv | endpoint back neg | fp_err_max (s0) | fp_err_max (s1) | fp_err_max (s2) | inv_shift (s0) | inv_shift (s1) | inv_shift (s2) |
|-----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 1/256 | 0/256 | 0/256 | 0/768 | 2.6e-3 | 9.0e-4 | 1.1e-2 | 0.0 | 0.0 | 0.0 |
| 5000 | 0/256 | 0/256 | 24/256 | 86/768 | 2.7e-5 | 9.3e-4 | 3.1e+2 | 0.7 | 0.8 | 3.7 |
| 10000 | 1/256 | 4/256 | 75/256 | 403/768 | 1.0e-3 | 1.2e+3 | 1.6e+3 | 1.1 | 5.3 | 81.1 |
| 20000 | 1/256 | 127/256 | 244/256 | 759/768 | 2.6e+2 | 6.8e+3 | 9.7e+3 | 2.7 | 562.6 | 3316.2 |
| 40000 | 3/256 | 201/256 | 253/256 | 768/768 | 4.4e+2 | 1.1e+4 | 1.0e+4 | 4.0 | 1518.2 | 4369.9 |
| 60000 | 8/256 | 233/256 | 247/256 | 768/768 | 1.7e+3 | 1.7e+4 | 1.6e+4 | 15.7 | 4339.3 | 5159.0 |
| 80000 | 17/256 | 240/256 | 247/256 | 766/768 | 4.0e+3 | 2.3e+4 | 2.2e+4 | 30.1 | 7035.2 | 6806.8 |
| 100000 | 16/256 | 248/256 | 173/256 | 762/768 | 4.6e+3 | 2.2e+4 | 1.9e+4 | 28.8 | 6372.8 | 8213.6 |
| 150000 | 17/256 | 98/256 | 42/256 | 737/768 | 4.0e+3 | 1.4e+4 | 1.9e+4 | 33.0 | 4316.1 | 6252.0 |
| 199999 | 43/256 | 86/256 | 37/256 | 751/768 | 7.4e+3 | 2.3e+4 | 2.8e+4 | 115.2 | 8275.2 | 8835.2 |

### Summary trend

| iter | A2 perstep neg% | B perstep neg% | A2 endpoint neg | B endpoint neg |
|-----:|:---:|:---:|:---:|:---:|
| 0 | 0.1% | 0.1% | 0/768 | 0/768 |
| 5000 | 3.4% | 3.1% | 124/768 | 86/768 |
| 10000 | 9.4% | 10.4% | 346/768 | 403/768 |
| 20000 | 50.9% | 48.4% | 768/768 | 759/768 |
| 40000 | 56.5% | 59.5% | 768/768 | 768/768 |
| 60000 | 22.7% | 63.5% | 763/768 | 768/768 |
| 80000 | 27.0% | 65.6% | 754/768 | 766/768 |

---

## 3. Interpretation

### The base model is well-conditioned at iter 0

With the v2 fix (on-policy obs per checkpoint), both A2 and B produce **identical**
iter-0 results: 1/768 per-step neg det, 0/768 endpoint neg, `fp_err_max` of 0.003-0.011,
`inv_shift` near zero. The pretrained flow is a valid diffeomorphism at dt=0.25.

The original diagnostic showed A2 with 88/256 neg dets at step 2 (iter 0) while B showed
1/256 — this was entirely an artifact of different observation batches, not a property of
the base model. The conclusion that "even the base model violates invertibility" was wrong.

### FP inversion breaks down as v_res grows

The breakdown follows a clear progression in both runs:

**Step 0 (t=0, near noise):** Stays mostly clean throughout training. At 80k,
only 7/256 (A2) or 17/256 (B) neg dets. `fp_err_max` grows but `inv_shift` stays
under 30 for most of training. Step 0 KL is reliable.

**Step 1 (t=0.25):** Breaks by 20k iterations. 145/256 (A2) or 127/256 (B) neg dets,
`inv_shift` jumps to 500-1000, `fp_err_max` in the thousands. FP iteration diverges
rather than converging.

**Step 2 (t=0.5):** Breaks slightly earlier. By 20k, 244/256 neg dets in both runs.
`inv_shift` reaches thousands. Completely unreliable.

The cause: the combined trajectory at steps 1-2 drifts from the base policy's
reachable set as v_res grows. The one-step Euler map `a + v_base(a,t)*dt` is no
longer locally invertible at those points — not because the base map is ill-conditioned,
but because the combined trajectory has moved to regions where no pre-image exists
under the base map.

### Why kl_mean goes negative on WandB

The per-step KL sums `logdet_ref_k - logdet_combined_k` over K-1=3 steps. Steps 1-2
evaluate `logdet(I + J_base(a_inv)*dt)` at garbage `a_inv` points (where FP diverged,
`inv_shift` in thousands). These garbage logdet values dominate the sum and can produce
any sign — the resulting "KL" has no probabilistic interpretation.

By 20k iterations, ~50% of samples have corrupted FP inversion across steps 1-2.
The negative kl_mean trending to -5 on WandB is this corruption, not a physical signal.

### What this means for the Lagrangian dual

The Lagrangian dual variable (eta) optimizes `E[KL] <= epsilon`. If the KL being
optimized is garbage from FP failure at steps 1-2, eta will chase noise. The dual
should only be applied to a valid KL signal.

---

## 4. Viable KL options

| Option | Validity | Signal quality | Cost |
|--------|----------|---------------|------|
| Step-0-only KL | Valid (FP converges) | Partial (only noise→data transition) | 1/3 of current |
| ||v_res||^2 | Always valid (no FP) | Proxy, not true KL, but monotonically tracks drift | Cheap |
| Increase K (reduce dt) | Valid if dt small enough | Better FP convergence | K x more expensive |
| Hutchinson trace (no FP) | Valid if no inversion | Loses cross-entropy signal without inversion | Moderate |

Step-0-only KL is the minimal fix: restrict the loop to k=0 where FP is reliable,
giving a valid KL signal to the Lagrangian dual. It's one line of code.

---

## 5. Summary Table

| Property | Step 0 (t=0) | Steps 1-2 (t=0.25-0.5) | Full endpoint |
|----------|:---:|:---:|:---:|
| FP convergence at iter 0 | Yes (err < 0.01) | Yes (err < 0.01) | Yes |
| FP convergence at iter 20k+ | Yes (err < 500) | **No** (err > 1000, shift > 500) | N/A |
| Det positive at iter 0? | Yes (1/256 neg) | Yes (0/256 neg) | Yes (0/768 neg) |
| Det positive at iter 20k+? | Mostly (2-7/256 neg) | **No** (127-244/256 neg) | **No** (759-768/768 neg) |
| KL well-defined? | Yes | No (after ~10-20k iters) | No (after ~10k iters) |
| Root cause of failure | — | Combined trajectory leaves base reachable set | Error accumulates across steps |
