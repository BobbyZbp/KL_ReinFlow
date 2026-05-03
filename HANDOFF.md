# Project Handoff — KL_ReinFlow

> **Status as of 2026-05-03**: code complete, mathematically verified, currently failing the empirical sanity check (KL term diverges on real trained base policy, reward stays below BC baseline). Ready for someone to debug the KL stability or reset the priors.

---

## 1. TL;DR

We are extending [ReinFlow](https://github.com/ReinFlow/ReinFlow) (NeurIPS 2025) — a PPO-based fine-tuner for flow-matching policies — by **replacing PPO with SAC + a residual velocity head + an exact marginal KL toward the pretrained base policy** (computed via the change-of-variables formula on a deterministic ODE).

The hope: ReinFlow's PPO can't compute a true marginal KL toward the base flow (its per-step noise injection breaks the change of variables), so it falls back to a Wasserstein-2 surrogate. By keeping the ODE deterministic except at the very last step, we recover the marginal density and get an *exact* KL — a tighter, more principled regularizer.

Code lives at https://github.com/BobbyZbp/KL_ReinFlow (fork of ReinFlow), branch `release`. Latest commit: `4350359`.

**The math is verified to be correct (smoke test).** The empirical run on a real pretrained Hopper checkpoint exposes a numerical issue: the backward fixed-point ODE solve fails to converge once the residual head pushes the θ-trajectory off the base flow's natural path, producing KL values of 1e17–1e19. Several defenses are now in place but have not been re-tested.

---

## 2. The Research Idea (one paragraph each)

**Source doc:** `New Project Idea 0501.md` (in user's working notes, not in repo)

**The architecture** (per [model/flow/ft_sac/sac_residual_flow.py](model/flow/ft_sac/sac_residual_flow.py)):

```
a^0 ~ N(0, I)
for k = 0..K-2:                                              # deterministic ODE
    a^{k+1} = a^k + (v_base + v_res)(a^k, t_k, s) * dt
eps ~ N(0, I)
a^K = a^{K-1} + v_base(a^{K-1}, t_{K-1}, s) * dt + sigma_phi(s) * eps
```

- `v_base` is the pretrained ReFlow policy (frozen, no grad)
- `v_res` is a small trainable residual MLP (zero-init last layer → policy starts identical to base)
- `sigma_phi(o)` is an observation-conditioned Gaussian std, bounded to [σ_min, σ_max] by sigmoid
- All noise lives at step `K-1`, so steps `0..K-2` form a deterministic diffeomorphism — change of variables applies, marginal density at `a^{K-1}` is exact.

**Loss** (off-policy SAC, no entropy bonus):

```
L_critic = (Q(s, a^K) - [r + γ(1-d) min Q'(s', a^{K}_next)])²       # twin Q + targets
L_actor  = -min(Q1, Q2)(s, a^K)  +  β · KL(p_θ‖p_base)  +  λ_J · ||∇v_res||_F²
```

The KL is computed at `a^{K-1}` (deterministic point) via:
- Forward: `log p_θ(a^{K-1}) = log p_0(x_0) - Σ log|det(I + (J_base+J_res)·dt)|` along the θ-trajectory
- Backward: invert the *base* ODE from `a^{K-1}_θ` via fixed-point iteration to recover `a^0_base`, then `log p_base = log p_0(a^0_base) - Σ log|det(I + J_base·dt)|` along the recovered base trajectory
- KL is `log p_θ - log p_base`, average over batch.

The Jacobian Frobenius regularizer keeps `v_res` small enough that the discrete map remains invertible (without it the ODE becomes stiff and the change of variables breaks down — this is the RNODE finding).

**Why no SAC entropy bonus?** The marginal entropy of `π_θ(a^K|s)` is intractable (would require integrating over the deterministic ODE pushforward convolved with the final Gaussian). We forgo it; exploration relies on bounded `σ_phi(s)` and the KL anchor toward the base.

---

## 3. What's in the Repo (relative to ReinFlow upstream)

| File | What |
|---|---|
| [model/flow/ft_sac/__init__.py](model/flow/ft_sac/__init__.py) | New package marker |
| [model/flow/ft_sac/sac_residual_flow.py](model/flow/ft_sac/sac_residual_flow.py) | **Main module.** `SACResidualFlow` (frozen `v_base`, trainable `v_res`, twin Q critic, target nets, `σ_φ` head). Methods: `sample_action` (reparameterized through ODE), `forward_with_logdet` (deterministic ODE + per-step Jacobians via `vmap(jacrev())`), `_backward_base_recover_trajectory` (fixed-point inversion), `backward_base_logprob` (Jacobians at inverted points), `compute_kl_and_action`, `loss_critic`, `loss_actor`. |
| [agent/finetune/reinflow/train_sac_residual_flow_agent.py](agent/finetune/reinflow/train_sac_residual_flow_agent.py) | Off-policy training loop (deque-based replay, twin Q + targets, delayed actor, target soft-update, grad clipping, running episode reward tracker). |
| [cfg/gym/finetune/hopper-v2/ft_sac_residual_flow_mlp.yaml](cfg/gym/finetune/hopper-v2/ft_sac_residual_flow_mlp.yaml) | Hopper config. K=4 (3 deterministic + 1 noisy), kl_weight=0.05, jac_weight=0.01, σ∈[0.05, 0.15], FP iters=30, actor_lr=3e-5, critic_lr=3e-4. |
| [script/test_sac_residual_flow.py](script/test_sac_residual_flow.py) | Smoke test — proves change-of-variables is mathematically correct via two analytical limits (v_res=0 and v_base=0 → KL must be exactly 0). |

**Not modified from ReinFlow upstream** — everything in `model/flow/` (FlowMLP), `model/common/critic.py` (CriticObsAct), `agent/finetune/train_agent.py` (parent class), env wrappers, etc. The new method is layered on top.

---

## 4. What's Verified

### 4.1 Smoke test (passes on both laptop GPU and cluster GPU)

```bash
python3 script/test_sac_residual_flow.py
```

Six checks pass:

1. `sample_action` — shapes correct, gradients route only to `v_res`/`σ_φ`/critic, never to `v_base`
2. `forward_with_logdet` — finite log_p_θ, correct shapes
3. `backward_base_logprob` — finite log_p_base, correct shapes
4. **a)** With `v_res ≡ 0` and random `v_base`: KL = `5.96e-8` (numerical zero) ✓ — proves that when residual is zero, the θ-ODE and base-ODE produce identical densities
5. **b)** With `v_base ≡ 0`: KL = exact `0.0` ✓ — sharp correctness check
6. Critic + actor losses produce gradients on the right parameters; `v_base` receives 0 gradients in every code path

### 4.2 Timing (B=256, K=4, D=12 on RTX 6000 Ada)

| Pass | Time |
|---|---|
| Critic step | ~10 ms |
| Actor step (KL + Jacobian reg + ODE backprop) | ~50 ms |
| Projected 200K-iter full run | ~1.9h pure GPU + env stepping overhead |

---

## 5. Pilot Run History (the empirical part — currently broken)

### 5.1 Run 1: System smoke (with pretrained Hopper checkpoint)

- Cluster: `monakhova-compute-01`, RTX 6000 Ada
- Config: `n_train_itr=2000`, `n_explore_steps=500`, `n_envs=8`, wandb offline
- Result: 2000 iters completed, system end-to-end works

**Issues found:**
- `avg episode reward - train = 0.0`, `num episode = 0` — turned out to be **logging bug**, not actual zero reward. With `n_steps=1` (one env step per outer iter), no episode ever completed within a single iter, and the original episode-summary code only counted within-iter completions.
- `loss_actor` showed sporadic spikes to 1e8, even 1e16 — early-training Q-target instability common to SAC, but warranting defensive grad clip.

**Fix (commit `e1335c9`):** added per-env running-reward accumulator that flushes to a deque on `done`, plus grad-norm clipping (critic 10.0, actor 1.0).

### 5.2 Run 2: Real reward visible, but KL diverged

- Same setup, after fix.
- **Good news:** reward tracking now works. Saw 200 completed episodes, `avg episode reward = 731.95` over the deque.
- **Bad news:** KL exploded.

```
actor/kl_max     3.62e+19      ← should be < 10
actor/loss_kl    1.41e+17      ← mean should be 0.5–5 nats
actor/loss_actor 7.07e+15      ← dominated by KL term
avg reward train  731.95       ← below Hopper BC baseline (~1500–2000)
loss - critic     15.10        ← still bounded, healthy
```

**Diagnosis** (`HANDOFF` author's hypothesis): the backward fixed-point iteration in `backward_base_logprob` fails to converge once `v_res` has drifted the θ-trajectory off the base flow's "natural path." For those out-of-distribution input points, FP can't invert the stiff trained `v_base`, the recovered `a^0_base` is garbage, `log p_base` becomes wildly wrong, and KL diverges.

This is exactly the failure mode the user's pi0.5 verification doc warned about. The smoke test passed because `v_res=0` kept the θ-trajectory ON the base trajectory; once training pushes `v_res` away from zero, that protection vanishes.

**Fix (commit `4350359`, applied but not yet re-tested):**
1. Clamp per-sample KL to `[-50, 50]` and `nan_to_num` non-finite values
2. Bump `backward_fp_iters` from 10 to 30
3. Result of KL clamp: even if the inversion fails for some samples, KL stays bounded; reward should at least track the BC baseline since v_res is anchored.

**Run 3 (with these fixes) has NOT been executed yet.** That's the immediate next step.

---

## 6. Current State / Open Questions

### What works
- Architecture is implemented and mathematically correct (smoke test passes)
- Off-policy SAC loop runs end-to-end on real env + real pretrained checkpoint
- Reward tracking, gradient routing, frozen-base contract all verified
- Grad clipping prevents Q-blowup catastrophes

### What's broken (or unconfirmed)
- **KL stability on stiff trained base flow** — Run 2 showed 1e17 KL; the clamp+more-FP fix is committed but unverified.
- **Reward below BC** — even at 731, well below BC baseline (~1500–2000). Could be a downstream effect of bad KL gradients pushing `v_res` in wrong direction. Should resolve once KL is bounded — but if it doesn't, we have a deeper problem.

### Hypotheses to test (in priority order)

1. **Run 3 with the committed clamp + 30 FP iters.** Most likely fix is here.
2. If KL is bounded but reward still below BC: `actor_lr=3e-5` may be too aggressive given the residual MLP is small. Try `1e-5`.
3. If KL is bounded but reward stuck at BC: `kl_weight=0.05` too high — try `0.01` to allow more residual movement.
4. If KL still spikes occasionally: maybe FP iter limit not enough, or use Anderson acceleration / Newton step instead of plain Picard iteration.
5. **Long-shot architectural concern:** the Jacobian Frobenius regularizer is on `v_res` only. The combined map `v_base + v_res` is what needs to be a diffeomorphism. If `v_base` itself has steep regions (which trained flows do), even a small `v_res` can push the combined map across a zero-determinant boundary. Switching the reg to `||v_base + v_res||_F²` along the trajectory might be safer (but adds backward Jacobian cost on `v_base`).

### Diagnostic logs to add (if Run 3 still fails)

- Per-sample `sign` from `slogdet` — if any goes ≤ 0, the discrete map stopped being a diffeomorphism, KL is meaningless for those samples
- FP convergence error: `||a_prev_fp_n - a_prev_fp_{n-1}||` at last iter — should be < 1e-3
- Per-sample reconstruction error: `||a_Km1 - forward_base(a^0_base)||` — should be tiny if FP converged

---

## 7. Reproduction Recipe

### 7.1 On a fresh GPU node

```bash
# Get a GPU node
salloc -p gpu-interactive --gres=gpu:nvidia_rtx_6000_ada_generation:1 \
       --cpus-per-task=4 --mem=64G -t 8:00:00

# Persistent session
tmux new -s pilot

# Setup
cd ~/ReinFlow
conda activate reinflow

# Confirm env vars (from ~/.bashrc)
echo $REINFLOW_DIR $REINFLOW_LOG_DIR $REINFLOW_DATA_DIR $REINFLOW_WANDB_ENTITY

# Verify GPU + smoke test passes
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 script/test_sac_residual_flow.py
```

Expected on smoke test: "ALL CHECKS PASSED", `device = cuda`, KL exactly 0 in 4a/4b.

### 7.2 Pull pretrained checkpoint (one-time)

All ReinFlow-related data lives under `~/ReinFlow/` (see UNICORN.md §3). HuggingFace downloads go to `~/ReinFlow/hf_cache/`.

```bash
python3 -c "
from huggingface_hub import hf_hub_download
import os
REPO = 'ReinFlow/ReinFlow-data-checkpoints-logs'
HF = os.path.expanduser('~/ReinFlow/hf_cache')
hf_hub_download(repo_id=REPO, repo_type='dataset',
    filename='log/log_gym_d4rl_pretrained/hopper-medium-v2/1-ReFlow/state_40.pt',
    local_dir=HF)
hf_hub_download(repo_id=REPO, repo_type='dataset',
    filename='data-offline/gym_d4rl/hopper-medium-v2/normalization.npz',
    local_dir=HF)
"
```

### 7.3 Run pilot

```bash
CKPT=~/ReinFlow/hf_cache/log/log_gym_d4rl_pretrained/hopper-medium-v2/1-ReFlow/state_40.pt
NORM=~/ReinFlow/hf_cache/data-offline/gym_d4rl/hopper-medium-v2/normalization.npz

python3 script/run.py \
    --config-path=../cfg/gym/finetune/hopper-v2 \
    --config-name=ft_sac_residual_flow_mlp \
    base_policy_path=$CKPT \
    normalization_path=$NORM \
    train.n_train_itr=2000 \
    train.n_explore_steps=500 \
    env.n_envs=8 \
    wandb.offline_mode=true
```

### 7.4 Full run (only after pilot looks healthy)

Same command but drop the overrides; expect 4–8 hours wall time. For a real overnight run, wrap it in `sbatch` (template in [UNICORN.md §10](UNICORN.md)).

---

## 8. Healthy-Run Checklist

Watch these during a run:

| Metric | Healthy range | Action if outside |
|---|---|---|
| `actor/kl_max` per iter | ≤ 10 | If 50+: clamp working but FP not converging — bump FP iters |
| `actor/loss_kl` mean | 0.5–5 nats | If < 0.1: KL anchor too strong, residual frozen → lower `kl_weight` |
| `actor/loss_jac` | < 1 | If > 1: residual destabilizing flow → bump `jac_weight` |
| `loss - critic` | 1–100, decreasing then stable | If diverging: lower `critic_lr` or check buffer normalization |
| `actor/q_mean` | rises monotonically (or plateaus) | If collapsing: actor over-correcting → lower `actor_lr` |
| `avg episode reward - train` | ≥ BC baseline (~1500 for Hopper), ideally rising | Below BC: KL or grad clip dominating — see above |
| GPU util (`nvidia-smi`) | 30–80% | Lower → CPU-bottlenecked on env step, raise `env.n_envs` |

---

## 9. Project File Map

```
~/ReinFlow/                                          # cluster path
├── model/
│   ├── flow/
│   │   ├── ft_sac/                                  # NEW: this project
│   │   │   ├── __init__.py
│   │   │   └── sac_residual_flow.py                 # the model
│   │   ├── ft_ppo/                                  # original ReinFlow PPO (untouched)
│   │   ├── ft_baselines/                            # FQL, etc. (untouched)
│   │   └── mlp_flow.py                              # FlowMLP base class (untouched)
│   ├── common/critic.py                             # CriticObsAct (twin Q, untouched)
│   └── rl/                                          # baselines we don't use
├── agent/finetune/
│   ├── reinflow/
│   │   ├── train_sac_residual_flow_agent.py         # NEW: this project's trainer
│   │   ├── train_ppo_flow_agent.py                  # original ReinFlow PPO trainer
│   │   └── buffer.py                                # PPO buffer (we use deques instead)
│   └── train_agent.py                               # parent class (requires cfg.train.n_steps)
├── cfg/gym/finetune/hopper-v2/
│   ├── ft_sac_residual_flow_mlp.yaml                # NEW: this project's config
│   └── ft_ppo_reflow_mlp.yaml                       # original (good reference for hyperparams)
├── script/
│   ├── run.py                                       # Hydra entry point
│   ├── test_sac_residual_flow.py                    # NEW: smoke test
│   └── set_path.sh                                  # env vars (REINFLOW_DIR, etc.)
├── HANDOFF.md                                       # this file
└── UNICORN.md                                       # cluster-specific cheatsheet
```

---

## 10. Commit History (this project, in order)

```
4350359  Clamp per-sample KL to +/-50; bump backward FP iters to 30      ← LATEST
e1335c9  Track episode rewards across iters; clip critic+actor grads
38b281f  Fix actor_replay_ratio=512 yielding actor_freq=0
61f5553  Fix sac_residual_flow yaml: add n_steps required by parent
61a60a5  Fix non-smooth clamping in deterministic ODE; add smoke test
2978662  Add SAC residual-flow fine-tuner with exact deterministic KL    ← INITIAL
```

`git log --oneline -6` to see in repo.

---

## 11. References

- **ReinFlow** (the upstream we're forking): https://github.com/ReinFlow/ReinFlow
  - Paper: "Fine-tuning Flow Matching Policies via Policy Gradient with Stochastic Markov Process Theorem", NeurIPS 2025
- **OGBench** (where I went looking for SAC implementation reference): https://github.com/seohongpark/ogbench
  - Paper: "OGBench: Benchmarking Offline Goal-Conditioned RL", ICLR 2025
  - Their SAC at `ogbench/impls/agents/sac.py` is for online expert data collection, not their benchmarked algorithm
- **FQL** (Flow Q-Learning, the closest existing flow+critic baseline): https://github.com/seohongpark/fql
  - Implementation in this repo at [model/flow/ft_baselines/fql.py](model/flow/ft_baselines/fql.py)
- **RNODE** (Jacobian regularization for continuous normalizing flows): Finlay et al., ICML 2020
- **Internal docs** (in user notes, not in repo):
  - `New Project Idea 0501.md` — original design
  - `pi0.5 forward-backward marginal probability.md` — verification of change-of-variables for stiffer flows; key finding that backward FP iter must be evaluated at the inverted point and 10 iters give ~0.5 nat error on a real trained model

---

## 12. Open Tasks for Whoever Picks This Up

In rough order of importance:

1. **Run pilot 3** with the committed KL clamp + 30 FP iters; verify KL stays bounded
2. If reward still ≤ BC: lower `actor_lr` to 1e-5, retry
3. If reward tracks BC but doesn't improve: lower `kl_weight` to 0.01–0.02, retry
4. Add diagnostic logging for (a) `slogdet` sign, (b) FP convergence error, (c) reconstruction error of `a^{K-1}` round-trip
5. Once a single seed shows reward rising above BC: launch 3-seed full run via `sbatch`
6. Compare against ReinFlow's published Hopper PPO numbers
7. (Stretch) port to Walker2d and HalfCheetah; same config swap
8. (Stretch) explore replacing Picard FP iteration with Anderson acceleration if convergence is the bottleneck
9. (Stretch) ablation: just the chain-SAC variant (use chain log-prob instead of marginal KL) as a stepping-stone baseline, to isolate "PPO→SAC" from "stochastic chain → deterministic ODE + final Gaussian"

Good luck.
