# MIT License
# Copyright (c) 2025 ReinFlow Authors
"""
Off-policy SAC fine-tuning for a flow-matching policy with a residual velocity head
plus exact deterministic KL via change of variables.

Replaces the PPO chain-logprob fine-tuner. Uses the same outer scaffolding as the
existing Gaussian SAC trainer (deque-based replay, twin Q + targets, delayed actor
updates) but the model itself is a SACResidualFlow.
"""
import os
import pickle
import logging
from collections import deque

import numpy as np
import torch
import wandb

from util.timer import Timer
from agent.finetune.train_agent import TrainAgent
from model.flow.ft_sac.sac_residual_flow import SACResidualFlow

log = logging.getLogger(__name__)


class TrainSACResidualFlowAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.model: SACResidualFlow

        self.gamma = cfg.train.gamma

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            list(self.model.v_res.parameters()) + list(self.model.sigma_head.parameters()),
            lr=cfg.train.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
        )

        # SAC hyperparameters
        self.target_ema_rate = cfg.train.target_ema_rate
        self.scale_reward_factor = cfg.train.scale_reward_factor
        self.batch_size = cfg.train.batch_size

        # update frequencies (assume single env, like flow_baselines/train_sac_agent.py).
        # max(1, ...) defends against bad config that would give freq=0 (modulo-by-zero).
        self.critic_update_freq = max(1, int(cfg.train.batch_size / cfg.train.critic_replay_ratio))
        self.actor_update_freq = max(1, int(cfg.train.batch_size / cfg.train.actor_replay_ratio))

        self.buffer_size = cfg.train.buffer_size
        self.n_eval_episode = cfg.train.n_eval_episode
        self.n_explore_steps = cfg.train.n_explore_steps

        # Entropy temperature (alpha)
        self.auto_entropy_tuning = cfg.train.get("auto_entropy_tuning", True)
        if self.auto_entropy_tuning:
            self.target_entropy = cfg.train.get(
                "target_entropy", -float(cfg.action_dim * cfg.act_steps)
            )
            init_alpha = cfg.train.get("init_alpha", 0.1)
            self.log_alpha = torch.tensor(
                np.log(init_alpha), dtype=torch.float32,
                device=self.device, requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=cfg.train.get("alpha_lr", 3e-4)
            )
            self.alpha = init_alpha
        else:
            self.alpha = cfg.train.get("alpha", 0.1)
        self.model.alpha = self.alpha
        self.kl_weight = cfg.train.get("kl_weight", self.model.kl_weight)
        self.jac_weight = cfg.train.get("jac_weight", self.model.jac_weight)
        self.sigma_entropy_weight = cfg.train.get("sigma_entropy_weight", self.model.sigma_entropy_weight)
        self.model.kl_weight = self.kl_weight
        self.model.jac_weight = self.jac_weight
        self.model.sigma_entropy_weight = self.sigma_entropy_weight
        self.critic_warmup_iters = cfg.train.get("critic_warmup_iters", 0)

        self.kl_mode = cfg.train.get("kl_mode", "none")
        self.model.kl_mode = self.kl_mode
        self.kl_reward_weight = cfg.train.get("kl_reward_weight", 0.0)
        self.vres_l2_weight = cfg.train.get("vres_l2_weight", 0.0)
        self.model.vres_l2_weight = self.vres_l2_weight
        self.hutchinson_samples = cfg.train.get("hutchinson_samples", 1)
        self.model.hutchinson_samples = self.hutchinson_samples
        self.perstep_fp_iters = cfg.train.get("perstep_fp_iters", 10)
        self.model.perstep_fp_iters = self.perstep_fp_iters

        self.ema_absorb_freq = cfg.train.get("ema_absorb_freq", 0)
        self.ema_absorb_tau = cfg.train.get("ema_absorb_tau", 0.01)
        self.actor_update_count = 0

        self.critic_grad_clip = cfg.train.get("critic_grad_clip", None)
        self.actor_grad_clip = cfg.train.get("actor_grad_clip", 1.0)

        # n_steps per outer iteration is set by parent TrainAgent.__init__ from cfg.train.n_steps.
        # Default to 1 to mimic standard SAC if parent didn't set it.
        if not hasattr(self, "n_steps"):
            self.n_steps = cfg.train.get("n_steps", 1)

        self.resume_path = cfg.get("resume_path", None)

        alpha_info = (
            f"auto_alpha(init={self.alpha:.3f} target_H={self.target_entropy:.1f})"
            if self.auto_entropy_tuning else f"fixed_alpha={self.alpha}"
        )
        log.info(
            f"SACResidualFlow trainer: gamma={self.gamma} tau={self.target_ema_rate} "
            f"batch={self.batch_size} critic_freq={self.critic_update_freq} "
            f"actor_freq={self.actor_update_freq} explore_steps={self.n_explore_steps} "
            f"{alpha_info} kl_w={self.kl_weight} jac_w={self.jac_weight} "
            f"critic_warmup={self.critic_warmup_iters} "
            f"kl_mode={self.kl_mode} kl_reward_w={self.kl_reward_weight} "
            f"vres_l2_w={self.vres_l2_weight} "
            f"ema_freq={self.ema_absorb_freq} ema_tau={self.ema_absorb_tau}"
        )

    # ----------------------------------------------------------- checkpointing
    def save_checkpoint(self, replay_buffers=None):
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_update_count": self.actor_update_count,
        }
        if self.auto_entropy_tuning:
            data["log_alpha"] = self.log_alpha.detach().cpu()
            data["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        if replay_buffers is not None:
            data["replay"] = {
                k: list(v) for k, v in replay_buffers.items()
            }
        path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
        torch.save(data, path)
        latest = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(data, latest)
        log.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        log.info(f"Resuming from {path}")
        data = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(data["model"], strict=False)
        if missing:
            log.warning(f"Missing keys in checkpoint (new params): {missing}")
        if unexpected:
            log.warning(f"Unexpected keys in checkpoint: {unexpected}")
        if any(k.startswith("ref_v_res.") for k in missing):
            import copy
            self.model.ref_v_res = copy.deepcopy(self.model.v_res).to(self.device)
            for p in self.model.ref_v_res.parameters():
                p.requires_grad = False
            self.model.ref_v_res.eval()
            log.info("Initialized ref_v_res from current v_res (not in checkpoint)")
        if "actor_optimizer" in data:
            self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        if "critic_optimizer" in data:
            self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        if self.auto_entropy_tuning and "log_alpha" in data:
            self.log_alpha.data.copy_(data["log_alpha"].to(self.device))
            self.alpha = self.log_alpha.exp().item()
            self.model.alpha = self.alpha
            if "alpha_optimizer" in data:
                self.alpha_optimizer.load_state_dict(data["alpha_optimizer"])
            log.info(f"Restored log_alpha={self.log_alpha.item():.4f} (alpha={self.alpha:.6f})")
        self.itr = data["itr"] + 1
        self.actor_update_count = data.get("actor_update_count", 0)
        self._resumed_replay = data.get("replay", None)
        log.info(f"Resumed at itr={self.itr} actor_updates={self.actor_update_count} "
                 f"(replay={'yes' if self._resumed_replay else 'no'})")

    # ----------------------------------------------------------------- run
    def run(self):
        if self.resume_path and os.path.isfile(self.resume_path):
            self.load_checkpoint(self.resume_path)

        # FIFO replay buffers
        obs_buffer = deque(maxlen=self.buffer_size)
        next_obs_buffer = deque(maxlen=self.buffer_size)
        action_buffer = deque(maxlen=self.buffer_size)
        reward_buffer = deque(maxlen=self.buffer_size)
        terminated_buffer = deque(maxlen=self.buffer_size)

        if hasattr(self, "_resumed_replay") and self._resumed_replay is not None:
            r = self._resumed_replay
            obs_buffer.extend(r["obs"])
            next_obs_buffer.extend(r["next_obs"])
            action_buffer.extend(r["action"])
            reward_buffer.extend(r["reward"])
            terminated_buffer.extend(r["terminated"])
            log.info(f"Restored replay buffer with {len(obs_buffer)} transitions")
            del self._resumed_replay

        # Running per-env episode reward accumulators + finished-episode log.
        # n_steps=1 means each iter only sees 1 env step, so we must accumulate
        # reward across iters and only flush on `done`.
        ep_running_reward = np.zeros(self.n_envs, dtype=np.float64)
        ep_running_len = np.zeros(self.n_envs, dtype=np.int64)
        completed_ep_rewards = deque(maxlen=200)
        completed_ep_lens = deque(maxlen=200)
        recent_kl_penalties = deque(maxlen=200)
        recent_kl_raw = deque(maxlen=200)
        recent_raw_rewards = deque(maxlen=200)
        recent_modified_rewards = deque(maxlen=200)

        timer = Timer()
        run_results = []
        cnt_train_step = 0
        prev_obs_venv = None
        done_venv = np.zeros((1, self.n_envs))
        loss_critic_val = 0.0
        loss_actor_val = 0.0
        last_actor_info = {}
        last_critic_info = {}

        while self.itr < self.n_train_itr:
            if self.itr % 1000 == 0:
                print(f"Iter {self.itr}/{self.n_train_itr}")

            # video paths
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            eval_mode = (
                self.itr % self.val_freq == 0
                and self.itr > self.n_explore_steps
                and not self.force_train
            )
            n_steps = self.n_steps if not eval_mode else int(1e5)
            self.model.eval() if eval_mode else self.model.train()
            # always keep frozen submodules in eval
            self.model.v_base.eval()
            self.model.ref_v_res.eval()

            # reset env at iteration start (eval, requested, or first iter)
            firsts_trajs = np.zeros((n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or prev_obs_venv is None:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv
            reward_trajs = np.zeros((n_steps, self.n_envs))

            cnt_episode = 0
            for step in range(n_steps):
                # action selection
                if self.itr < self.n_explore_steps:
                    action_venv = self.venv.action_space.sample()
                else:
                    with torch.no_grad():
                        cond = {
                            "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                        }
                        a = self.model.sample_action(cond, deterministic=eval_mode)
                        action_venv = a.cpu().numpy()[:, : self.act_steps]
                action_venv = np.nan_to_num(action_venv, nan=0.0)
                action_venv = np.clip(action_venv, -1.0, 1.0)

                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                reward_trajs[step] = reward_venv
                firsts_trajs[step + 1] = done_venv

                if not eval_mode:
                    kl_penalty_np = np.zeros(self.n_envs)
                    if (
                        self.kl_mode == "reward_penalty"
                        and self.kl_reward_weight > 0
                        and self.itr >= self.n_explore_steps
                    ):
                        with torch.no_grad():
                            cond_kl = {
                                "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                            }
                            kl_vals = self.model.compute_kl_for_reward(cond_kl)
                            kl_penalty_np = kl_vals.cpu().numpy()

                    for i in range(self.n_envs):
                        s_prev = prev_obs_venv["state"][i]
                        if "final_obs" in info_venv[i]:
                            s_next = info_venv[i]["final_obs"]["state"]
                        else:
                            s_next = obs_venv["state"][i]
                        if np.any(np.isnan(s_prev)) or np.any(np.isnan(s_next)):
                            continue
                        obs_buffer.append(s_prev)
                        next_obs_buffer.append(s_next)
                        action_buffer.append(action_venv[i])
                        r_raw = float(reward_venv[i] * self.scale_reward_factor)
                        kl_pen = self.kl_reward_weight * float(kl_penalty_np[i])
                        r_modified = r_raw - kl_pen
                        reward_buffer.append(r_modified)
                        terminated_buffer.append(float(terminated_venv[i]))
                        recent_raw_rewards.append(r_raw)
                        recent_modified_rewards.append(r_modified)
                        recent_kl_penalties.append(kl_pen)
                        recent_kl_raw.append(float(kl_penalty_np[i]))

                    # Running episode reward / length, accumulated across iters.
                    # Flush to completed_* deques whenever an env terminates or truncates.
                    ep_running_reward += reward_venv
                    ep_running_len += 1
                    for i in range(self.n_envs):
                        if done_venv[i]:
                            completed_ep_rewards.append(float(ep_running_reward[i]))
                            completed_ep_lens.append(int(ep_running_len[i]))
                            ep_running_reward[i] = 0.0
                            ep_running_len[i] = 0

                prev_obs_venv = obs_venv
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

                cnt_episode += np.sum(done_venv)
                if eval_mode and cnt_episode >= self.n_eval_episode:
                    break

            # episode reward summary
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    s, e = env_steps[i], env_steps[i + 1]
                    if e - s > 1:
                        episodes_start_end.append((env_ind, s, e - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[s : e + 1, env_ind] for env_ind, s, e in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array([np.sum(r) for r in reward_trajs_split])
                episode_best_reward = np.array(
                    [np.max(r) / self.act_steps for r in reward_trajs_split]
                )
                avg_episode_reward = float(np.mean(episode_reward))
                avg_best_reward = float(np.mean(episode_best_reward))
                success_rate = float(
                    np.mean(episode_best_reward >= self.best_reward_threshold_for_success)
                )
            else:
                # Fall back to the running accumulator (mean over recent finished episodes).
                if len(completed_ep_rewards) > 0:
                    num_episode_finished = len(completed_ep_rewards)
                    avg_episode_reward = float(np.mean(completed_ep_rewards))
                    avg_best_reward = avg_episode_reward
                    success_rate = float(
                        np.mean(np.array(completed_ep_rewards) >= self.best_reward_threshold_for_success)
                    )
                else:
                    num_episode_finished = 0
                    avg_episode_reward = 0.0
                    avg_best_reward = 0.0
                    success_rate = 0.0

            # =================================================== updates
            if (
                not eval_mode
                and self.itr > self.n_explore_steps
                and self.itr % self.critic_update_freq == 0
                and len(obs_buffer) >= self.batch_size
            ):
                inds = np.random.choice(len(obs_buffer), self.batch_size, replace=False)
                obs_b = torch.from_numpy(np.stack([obs_buffer[i] for i in inds])).float().to(self.device)
                next_obs_b = torch.from_numpy(np.stack([next_obs_buffer[i] for i in inds])).float().to(self.device)
                actions_b = torch.from_numpy(np.stack([action_buffer[i] for i in inds])).float().to(self.device)
                rewards_b = torch.from_numpy(np.array([reward_buffer[i] for i in inds])).float().to(self.device)
                terminated_b = torch.from_numpy(np.array([terminated_buffer[i] for i in inds])).float().to(self.device)

                # The buffer stores act_steps-truncated actions; pad to (Ta, Da) for the critic.
                # In ReinFlow's setup act_steps == horizon_steps, so this is usually a no-op.
                if actions_b.dim() == 2:
                    # was (B, Da) from a single-step action env; reshape to (B, 1, Da)
                    actions_b = actions_b.unsqueeze(1)
                if actions_b.shape[1] < self.model.horizon_steps:
                    pad = torch.zeros(
                        actions_b.shape[0],
                        self.model.horizon_steps - actions_b.shape[1],
                        actions_b.shape[2],
                        device=self.device,
                    )
                    actions_b = torch.cat([actions_b, pad], dim=1)

                obs_dict = {"state": obs_b}
                next_obs_dict = {"state": next_obs_b}

                # ---- critic step
                loss_critic, last_critic_info = self.model.loss_critic(
                    obs_dict, next_obs_dict, actions_b, rewards_b, terminated_b, self.gamma
                )
                if not torch.isfinite(loss_critic):
                    log.error(f"iter {self.itr}: critic loss is {loss_critic.item()}, skipping update")
                    log.error(f"  critic info: {last_critic_info}")
                    log.error(f"  rewards_b: min={rewards_b.min():.4f} max={rewards_b.max():.4f} "
                              f"has_nan={torch.isnan(rewards_b).any()}")
                else:
                    self.critic_optimizer.zero_grad()
                    loss_critic.backward()
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.critic.parameters(),
                        max_norm=self.critic_grad_clip if self.critic_grad_clip else float('inf'),
                    )
                    self.critic_optimizer.step()
                    self.model.update_target_critic(self.target_ema_rate)
                    last_critic_info["grad_norm"] = critic_grad_norm.item()
                loss_critic_val = loss_critic.item()

                # ---- actor step (delayed; skipped during critic warmup)
                past_warmup = self.itr >= self.n_explore_steps + self.critic_warmup_iters
                if past_warmup and self.itr % self.actor_update_freq == 0:
                    loss_actor, last_actor_info = self.model.loss_actor(obs_dict)
                    if not torch.isfinite(loss_actor):
                        log.error(f"iter {self.itr}: actor loss is {loss_actor.item()}, skipping update")
                        log.error(f"  actor info: {last_actor_info}")
                        nan_components = {k: v for k, v in last_actor_info.items()
                                          if isinstance(v, float) and (v != v or abs(v) > 1e15)}
                        if nan_components:
                            log.error(f"  NaN/huge components: {nan_components}")
                    else:
                        self.actor_optimizer.zero_grad()
                        loss_actor.backward()
                        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                            list(self.model.v_res.parameters()) + list(self.model.sigma_head.parameters()),
                            max_norm=self.actor_grad_clip if self.actor_grad_clip else float('inf'),
                        )
                        self.actor_optimizer.step()
                        last_actor_info["grad_norm"] = actor_grad_norm.item()

                        # Auto-tune alpha (SB3-style: gradient w.r.t. log_alpha is -(logp + H_target))
                        if self.auto_entropy_tuning:
                            log_prob_val = last_actor_info.get("log_prob_mean", 0.0)
                            alpha_loss = -(self.log_alpha * (log_prob_val + self.target_entropy))
                            self.alpha_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.alpha_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()
                            self.model.alpha = self.alpha
                            last_actor_info["alpha"] = self.alpha
                            last_actor_info["alpha_loss"] = alpha_loss.item()

                        self.actor_update_count += 1
                        if self.ema_absorb_freq > 0 and self.actor_update_count % self.ema_absorb_freq == 0:
                            if self.kl_mode.endswith("_ref"):
                                self.model.update_reference_policy(self.ema_absorb_tau)
                                last_actor_info["ref_ema_updated"] = 1.0
                            else:
                                try:
                                    self.model.absorb_residual_into_base(self.ema_absorb_tau)
                                    last_actor_info["ema_absorbed"] = 1.0
                                except RuntimeError as e:
                                    if self.actor_update_count == self.ema_absorb_freq:
                                        log.error(f"EMA absorption failed (disabling): {e}")
                                        self.ema_absorb_freq = 0

                    loss_actor_val = loss_actor.item()
                    with torch.no_grad():
                        vres_norm = sum(p.norm().item()**2 for p in self.model.v_res.parameters())**0.5
                    last_actor_info["v_res_param_norm"] = vres_norm

            # save checkpoint (model + optimizers + replay buffer for resume)
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_checkpoint(replay_buffers={
                    "obs": obs_buffer,
                    "next_obs": next_obs_buffer,
                    "action": action_buffer,
                    "reward": reward_buffer,
                    "terminated": terminated_buffer,
                })

            # log
            run_results.append({"itr": self.itr, "step": cnt_train_step})
            if self.itr % self.log_freq == 0 and self.itr > self.n_explore_steps:
                t = timer()
                if eval_mode:
                    log.info(
                        f"eval | success {success_rate:.3f} | reward {avg_episode_reward:.3f} | "
                        f"best {avg_best_reward:.3f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | actor {loss_actor_val:8.4f} | "
                        f"critic {loss_critic_val:8.4f} | reward {avg_episode_reward:8.4f} | t {t:6.2f}"
                    )
                    if self.use_wandb:
                        wandb_log_dict = {
                            "total env step": cnt_train_step,
                            "loss - critic": loss_critic_val,
                            "loss - actor": loss_actor_val,
                            "avg episode reward - train": avg_episode_reward,
                            "num episode - train": num_episode_finished,
                            "buffer_size": len(obs_buffer),
                            **{f"actor/{k}": v for k, v in last_actor_info.items()},
                            **{f"critic/{k}": v for k, v in last_critic_info.items()},
                        }
                        if len(completed_ep_rewards) > 0:
                            wandb_log_dict["episode/reward_mean"] = float(np.mean(completed_ep_rewards))
                            wandb_log_dict["episode/reward_std"] = float(np.std(completed_ep_rewards))
                            wandb_log_dict["episode/reward_max"] = float(np.max(completed_ep_rewards))
                            wandb_log_dict["episode/reward_min"] = float(np.min(completed_ep_rewards))
                            wandb_log_dict["episode/length_mean"] = float(np.mean(completed_ep_lens))
                        if len(recent_raw_rewards) > 0:
                            wandb_log_dict["reward/raw_mean"] = float(np.mean(recent_raw_rewards))
                            wandb_log_dict["reward/modified_mean"] = float(np.mean(recent_modified_rewards))
                            wandb_log_dict["reward/kl_penalty_mean"] = float(np.mean(recent_kl_penalties))
                        if len(recent_kl_raw) > 0:
                            wandb_log_dict["reward/kl_raw_mean"] = float(np.mean(recent_kl_raw))
                            wandb_log_dict["reward/kl_raw_max"] = float(np.max(recent_kl_raw))
                            wandb_log_dict["reward/kl_raw_min"] = float(np.min(recent_kl_raw))
                        wandb.log(wandb_log_dict, step=self.itr, commit=True)
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
