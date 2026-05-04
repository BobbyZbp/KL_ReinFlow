| Replace | With | Notes |
| --- | --- | --- |
| NoisyFlowMLP | `ResidualFlowMLP` | Frozen `v_base = FlowMLP` (pretrained, no grad). Small `v_res` MLP, same `(action, t, cond) → vel` signature. Observation-conditioned `σ_φ(o)` head (single output, sigmoid into `[σ_min, σ_max]`) — **only consumed at step K-1**. |
| PPOFlow | `SACResidualFlow` (`model/flow/ft_sac/sac_residual_flow.py`) | Owns `v_base` (frozen), `v_res`, `σ_φ`, double-Q `CriticObsAct`, target critics. **No `actor_old` for IS purposes** — base is the reference for KL, no PPO ratio. |
| PPOFlowBuffer | `SACFlowBuffer` | Stores `(s, a^K, r, s', done)` only. Drop chains, advantages, returns. Pattern from train_sac_agent.py:170-188. |
| TrainPPOFlowAgent | `TrainSACResidualFlowAgent` | Off-policy loop. |

## Files added / changed

| # | Path | What | Why |
| --- | --- | --- | --- |
| 1 | model/flow/ft_sac/**init**.py | New empty package marker | Mirrors `model/flow/ft_ppo/`, `model/flow/ft_baselines/`. Lets Hydra resolve `_target_: model.flow.ft_sac....` |
| 2 | model/flow/ft_sac/sac_residual_flow.py | **`SACResidualFlow`** + **`SigmaHead`**. Frozen `v_base` (loaded from pretrained ReFlow checkpoint, `requires_grad=False`), trainable `v_res` (zero-init last layer so the policy starts identical to base), twin Q-critic `CriticObsAct` with target net, observation-conditioned `sigma_phi(o)` bounded to `[sigma_min, sigma_max]` via sigmoid. Methods: `sample_action` (reparameterized — *no* `@torch.no_grad`), `forward_with_logdet` (deterministic ODE for K-1 steps with `vmap(jacrev(...))` per-step Jacobians + `slogdet`), `_backward_base_recover_trajectory` (10-step fixed-point iteration to invert the base ODE), `backward_base_logprob` (Jacobians evaluated at *inverted* points — the trap the pi0.5 doc flags), `compute_kl_and_action`, `compute_jacobian_reg` (Frobenius on `v_res` only), `loss_critic` (no entropy term in target), `loss_actor` (`-min(Q1,Q2) + β·KL + λ_J·jac_reg`). | Replaces `PPOFlow`. Embodies the proposal: deterministic ODE → exact marginal KL via change of variables; SAC reparameterization → no chain log-prob, no IS ratio; residual head → small fine-tuning footprint; Jacobian reg → flow stays a diffeomorphism. |
| 3 | agent/finetune/reinflow/train_sac_residual_flow_agent.py | **`TrainSACResidualFlowAgent`**. Off-policy loop with `deque`-based replay (same pattern as the existing Gaussian-SAC trainer), random-action warmup for `n_explore_steps`, critic update every `critic_replay_ratio` steps, delayed actor update every `actor_replay_ratio` steps, target soft-update after each critic step, twin optimizer (actor optimizes `v_res ∪ sigma_head` only; `v_base` stays frozen), wandb logging. | Replaces `TrainPPOFlowAgent`. Off-policy buys back sample efficiency (each K-step rollout is expensive) and matches SAC semantics. |
| 4 | cfg/gym/finetune/hopper-v2/ft_sac_residual_flow_mlp.yaml | Hopper config: `denoising_steps=4` (3 deterministic + 1 noisy), `actor_lr=3e-5` (small per idea doc §11), `critic_lr=3e-4`, `target_ema_rate=0.005`, `kl_weight=0.05`, `jac_weight=0.01`, `sigma_min=0.05`, `sigma_max=0.15`, `backward_fp_iters=10` (per pi0.5 verification → <0.15% log-prob bias), `n_explore_steps=5000`, replay `1e6`. Critic switched to `CriticObsAct` (double-Q on `(s, a^K)`); residual MLP is small (`[128, 128]`) vs base `[512, 512, 512]`. | Concrete drop-in for the existing PPO ReFlow Hopper config; same env, base checkpoint, normalization. |

## What was deleted from the PPO design (and why)

- **Per-step Gaussian noise** (`NoisyFlowMLP`, `ExploreNoiseNet`, `min/max_logprob_denoising_std`, `learn_decay`, `const_schedule_itr`) — gone. Noise lives only at step `K-1`, where it doesn't enter the density. This is what restores ODE structure for the change of variables.
- **Joint chain log-prob** (`get_logprobs`, `logprob_min/max` clamps, `logprob_steps` normalization, `account_for_initial_stochasticity`) — gone. SAC reparam doesn't need any log-prob.
- **PPO-specific terms** (importance ratio, `clip_ploss_coef*`, `clip_vloss_coef`, `approx_kl`, `clipfrac`, `update_epochs`, `gae_lambda`, advantage normalization, `target_kl`) — gone.
- **W2 BC loss** — replaced by exact `D_KL(p_θ || p_base)` at `a^{K-1}`.
- **State-only `V(s)` critic** — replaced by `Q(s, a)` double critic on the executed action.
- **`actor_old` for IS** — gone. The reference for KL is `v_base` itself (frozen).
- **`log_alpha` / temperature** — *not* introduced. The proposal explicitly forgoes the SAC entropy term (marginal entropy is intractable here); exploration leans on bounded `sigma_phi` and the KL-to-base.

## What was kept from ReinFlow

- Pretrained checkpoint loader (`ema` preferred), `act_min/max` and `denoised_clip_value` clamps, `randn_clip_value` on the final ε, env scaffolding from `TrainAgent`, the `act_steps` chunking, wandb hooks.

### Bug found and fixed

`forward_with_logdet` and `sample_action` were clamping intermediate ODE states to `[-denoised_clip_value, denoised_clip_value]`, copied from the PPO codepath. Clamping is non-smooth and breaks the change-of-variables formula — the discrete map stops being a diffeomorphism wherever clamping bites. Initial KL probe failed because the forward θ-trajectory's `x_0` was getting saturated while the backward solve recovered the true (pre-clamp) start point.

Fixed in model/flow/ft_sac/sac_residual_flow.py:

- Removed `a.clamp(...)` inside `sample_action`'s deterministic ODE loop
- Removed `a.clamp(...)` inside `forward_with_logdet`'s loop
- Kept the final `a_K.clamp(act_min, act_max)` (env safety, only on the executed action)

The Jacobian Frobenius regularization (λ_J term) is what keeps the deterministic flow well-conditioned now, exactly as the proposal intended.