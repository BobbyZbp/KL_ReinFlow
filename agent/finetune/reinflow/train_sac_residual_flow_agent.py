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

        self.alpha = cfg.train.get("alpha", self.model.alpha)
        self.model.alpha = self.alpha
        self.kl_weight = cfg.train.get("kl_weight", self.model.kl_weight)
        self.jac_weight = cfg.train.get("jac_weight", self.model.jac_weight)
        self.sigma_entropy_weight = cfg.train.get("sigma_entropy_weight", self.model.sigma_entropy_weight)
        self.model.kl_weight = self.kl_weight
        self.model.jac_weight = self.jac_weight
        self.model.sigma_entropy_weight = self.sigma_entropy_weight
        self.critic_warmup_iters = cfg.train.get("critic_warmup_iters", 0)

        # n_steps per outer iteration is set by parent TrainAgent.__init__ from cfg.train.n_steps.
        # Default to 1 to mimic standard SAC if parent didn't set it.
        if not hasattr(self, "n_steps"):
            self.n_steps = cfg.train.get("n_steps", 1)

        log.info(
            f"SACResidualFlow trainer: gamma={self.gamma} tau={self.target_ema_rate} "
            f"batch={self.batch_size} critic_freq={self.critic_update_freq} "
            f"actor_freq={self.actor_update_freq} explore_steps={self.n_explore_steps} "
            f"alpha={self.alpha} kl_w={self.kl_weight} jac_w={self.jac_weight} "
            f"sigma_ent_w={self.sigma_entropy_weight} critic_warmup={self.critic_warmup_iters}"
        )

    # ----------------------------------------------------------------- run
    def run(self):
        # FIFO replay buffers
        obs_buffer = deque(maxlen=self.buffer_size)
        next_obs_buffer = deque(maxlen=self.buffer_size)
        action_buffer = deque(maxlen=self.buffer_size)
        reward_buffer = deque(maxlen=self.buffer_size)
        terminated_buffer = deque(maxlen=self.buffer_size)

        # Running per-env episode reward accumulators + finished-episode log.
        # n_steps=1 means each iter only sees 1 env step, so we must accumulate
        # reward across iters and only flush on `done`.
        ep_running_reward = np.zeros(self.n_envs, dtype=np.float64)
        ep_running_len = np.zeros(self.n_envs, dtype=np.int64)
        completed_ep_rewards = deque(maxlen=200)
        completed_ep_lens = deque(maxlen=200)

        timer = Timer()
        run_results = []
        cnt_train_step = 0
        done_venv = np.zeros((1, self.n_envs))
        loss_critic_val = 0.0
        loss_actor_val = 0.0
        last_actor_info = {}

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
            # always keep base policy in eval (frozen)
            self.model.v_base.eval()

            # reset env at iteration start (eval, requested, or first iter)
            firsts_trajs = np.zeros((n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or self.itr == 0:
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

                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                reward_trajs[step] = reward_venv
                firsts_trajs[step + 1] = done_venv

                if not eval_mode:
                    for i in range(self.n_envs):
                        obs_buffer.append(prev_obs_venv["state"][i])
                        if "final_obs" in info_venv[i]:
                            next_obs_buffer.append(info_venv[i]["final_obs"]["state"])
                        else:
                            next_obs_buffer.append(obs_venv["state"][i])
                        action_buffer.append(action_venv[i])
                    reward_buffer.extend((reward_venv * self.scale_reward_factor).tolist())
                    terminated_buffer.extend(terminated_venv.tolist())

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
                loss_critic = self.model.loss_critic(
                    obs_dict, next_obs_dict, actions_b, rewards_b, terminated_b, self.gamma
                )
                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                # Clip critic gradients (defense against early-training Q-target divergence)
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=10.0)
                self.critic_optimizer.step()
                self.model.update_target_critic(self.target_ema_rate)
                loss_critic_val = loss_critic.item()

                # ---- actor step (delayed; skipped during critic warmup)
                past_warmup = self.itr >= self.n_explore_steps + self.critic_warmup_iters
                if past_warmup and self.itr % self.actor_update_freq == 0:
                    loss_actor, last_actor_info = self.model.loss_actor(obs_dict)
                    self.actor_optimizer.zero_grad()
                    loss_actor.backward()
                    # Clip actor gradients (the gradient chains through K-1 ODE Jacobians;
                    # without this, occasional bad samples can produce huge updates)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.v_res.parameters()) + list(self.model.sigma_head.parameters()),
                        max_norm=1.0,
                    )
                    self.actor_optimizer.step()
                    loss_actor_val = loss_actor.item()

            # save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

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
                            **{f"actor/{k}": v for k, v in last_actor_info.items()},
                        }
                        wandb.log(wandb_log_dict, step=self.itr, commit=True)
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
