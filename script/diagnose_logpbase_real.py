"""
Decompose log_p_base using REAL observations from policy rollouts in Hopper.

Collects observations by running the trained policy in the environment,
then analyzes FP inversion accuracy and log_p_base decomposition.

Usage:
    python script/diagnose_logpbase_real.py --run A --iters 0,50000,100000,199999
"""
import argparse
import sys
import os
import math

import torch
import numpy as np
from torch.func import jacrev, vmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.flow.mlp_flow import FlowMLP
from model.flow.ft_sac.sac_residual_flow import SACResidualFlow, SigmaHead
from model.common.critic import CriticObsAct


BASE_POLICY_PATH = "/home/tw559/steering_pi/KL_ReinFlow/log/gym/pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_D4RL_42/state_40.pt"
NORM_PATH = "/home/tw559/steering_pi/KL_ReinFlow/hf_cache/data-offline/gym/hopper-medium-v2/normalization.npz"


def build_model(device):
    obs_dim, action_dim, horizon_steps, cond_dim = 11, 3, 4, 11
    base_policy = FlowMLP(
        horizon_steps=horizon_steps, action_dim=action_dim, cond_dim=cond_dim,
        time_dim=16, mlp_dims=[512, 512, 512], activation_type="ReLU",
        out_activation_type="Identity", use_layernorm=False, residual_style=True)
    residual_policy = FlowMLP(
        horizon_steps=horizon_steps, action_dim=action_dim, cond_dim=cond_dim,
        time_dim=16, mlp_dims=[128, 128], activation_type="ReLU",
        out_activation_type="Identity", use_layernorm=False, residual_style=False)
    critic = CriticObsAct(
        cond_dim=cond_dim, mlp_dims=[256, 256, 256], action_dim=action_dim,
        action_steps=horizon_steps, activation_type="Mish",
        use_layernorm=True, residual_tyle=False, double_q=True)
    sigma_head = SigmaHead(
        cond_dim=cond_dim, action_dim=action_dim, horizon_steps=horizon_steps,
        hidden_dims=[64, 64], sigma_min=0.05, sigma_max=0.15)
    model = SACResidualFlow(
        device=device, base_policy=base_policy, residual_policy=residual_policy,
        critic=critic, sigma_head=sigma_head, actor_policy_path=BASE_POLICY_PATH,
        act_dim=action_dim, horizon_steps=horizon_steps, act_min=-1, act_max=1,
        obs_dim=obs_dim, cond_steps=1, inference_steps=4, backward_fp_iters=30,
        alpha=0.1, kl_weight=0.05, jac_weight=0.0, sigma_entropy_weight=0.1,
        zero_init_residual=True, kl_mode="none")
    return model


def collect_obs_from_env(model, device, n_obs=256, max_ep_steps=1000):
    """Roll out the policy in Hopper to collect real observations."""
    import gym
    import d4rl.gym_mujoco

    norm = np.load(NORM_PATH)
    obs_min = norm["obs_min"]
    obs_max = norm["obs_max"]

    env = gym.make("hopper-medium-v2")
    collected = []
    obs = env.reset()

    while len(collected) < n_obs:
        # Normalize obs same way as training
        obs_norm = -1.0 + 2.0 * (obs - obs_min) / (obs_max - obs_min + 1e-8)
        obs_norm = np.clip(obs_norm, -1.0, 1.0)
        collected.append(obs_norm.copy())

        # Step with trained policy
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
            cond = {"state": obs_t}
            a = model.sample_action(cond, deterministic=False)
            action = a[0, :1].cpu().numpy()  # act_steps=1 from multi_step wrapper? use first step
            action = np.clip(action, -1.0, 1.0)

        obs, reward, done, info = env.step(action.flatten()[:3])
        if done:
            obs = env.reset()

    env.close()
    return np.array(collected[:n_obs])


@torch.no_grad()
def decompose_logpbase(model, obs_batch, z_batch, device, label=""):
    """Full decomposition of log_p_base and FP accuracy check."""
    B = z_batch.shape[0]
    K = model.inference_steps
    dt = 1.0 / K
    D = model.act_dim_total
    I_D = torch.eye(D, device=device).unsqueeze(0)
    cond = {"state": obs_batch}

    # Forward combined ODE
    a_theta = z_batch.clone()
    for k in range(K - 1):
        t = torch.full((B,), k * dt, device=device)
        v = model._combined_velocity(a_theta, t, cond)
        a_theta = a_theta + v * dt

    # Forward base ODE (same noise)
    a_base = z_batch.clone()
    for k in range(K - 1):
        t = torch.full((B,), k * dt, device=device)
        v = model.v_base(a_base, t, cond)
        a_base = a_base + v * dt

    endpoint_div = (a_theta - a_base).view(B, -1).norm(dim=-1)

    print(f"\n  === {label} ===")
    print(f"  Endpoint divergence: mean={endpoint_div.mean():.4f}  max={endpoint_div.max():.4f}")

    for target_name, a_Km1 in [("THETA endpoint", a_theta), ("BASE endpoint", a_base)]:
        print(f"\n  --- FP inversion at {target_name} ---")

        # FP backward recovery
        traj_nograd, traj_t = model._backward_base_recover_trajectory(a_Km1, cond)
        a0 = traj_nograd[0]

        has_nan = torch.isnan(a0).any(dim=(-2, -1))
        has_inf = torch.isinf(a0).any(dim=(-2, -1))
        valid = ~(has_nan | has_inf)
        n_valid = valid.sum().item()
        print(f"  a^0 recovered: valid={n_valid}/{B}  nan={has_nan.sum().item()}  inf={has_inf.sum().item()}")

        if n_valid == 0:
            print(f"  ALL SAMPLES DIVERGED — FP inversion completely failed")
            continue

        # Filter to valid samples
        a0_v = a0[valid]
        a_Km1_v = a_Km1[valid]
        cond_v = {"state": obs_batch[valid]}
        traj_v = [t[valid] for t in traj_nograd]
        traj_t_v = [t[valid] for t in traj_t]

        # Round-trip: forward from recovered a^0
        a_fwd = a0_v.clone()
        for k in range(K - 1):
            t = torch.full((n_valid,), k * dt, device=device)
            v = model.v_base(a_fwd, t, cond_v)
            a_fwd = a_fwd + v * dt
        rt_err = (a_fwd - a_Km1_v).view(n_valid, -1).norm(dim=-1)

        a0_norm = a0_v.view(n_valid, -1).norm(dim=-1)
        log_p0 = -0.5 * (a0_v ** 2).sum(dim=(-2, -1)) - 0.5 * D * math.log(2 * math.pi)

        # Per-step logdet
        sum_logdet = torch.zeros(n_valid, device=device)
        for k in range(K - 1):
            a_k = traj_v[k]
            t_k = traj_t_v[k]
            J = vmap(jacrev(model._per_sample_base_velocity_flat, argnums=0))(
                a_k.view(n_valid, D), t_k, cond_v["state"])
            M = I_D + J * dt
            sign, logabsdet = torch.linalg.slogdet(M)
            neg_sign = (sign < 0).sum().item()
            sum_logdet = sum_logdet + logabsdet
            print(f"    step {k} logdet: mean={logabsdet.mean():.4f}  min={logabsdet.min():.4f}  max={logabsdet.max():.4f}  neg_sign={neg_sign}/{n_valid}")

        log_p_base = log_p0 - sum_logdet
        pct_p0 = abs(log_p0.mean().item()) / (abs(log_p0.mean().item()) + abs(sum_logdet.mean().item()) + 1e-30) * 100

        print(f"  Round-trip error:   mean={rt_err.mean():.6f}  max={rt_err.max():.6f}")
        print(f"  ||a^0|| recovered:  mean={a0_norm.mean():.4f}  max={a0_norm.max():.4f}")
        print(f"  log_p_0(a^0):       mean={log_p0.mean():.2f}  min={log_p0.min():.2f}")
        print(f"  sum_logdet:         mean={sum_logdet.mean():.2f}  min={sum_logdet.min():.2f}")
        print(f"  log_p_base:         mean={log_p_base.mean():.2f}  min={log_p_base.min():.2f}  std={log_p_base.std():.2f}")
        print(f"  Contribution: log_p_0 = {pct_p0:.1f}% of |log_p_base|")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="A")
    parser.add_argument("--iters", type=str, default="0,20000,50000,100000,199999")
    parser.add_argument("--n_obs", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_map = {"A": "ablation_runA_baseline", "B": "ablation_runB_reward_penalty",
               "C": "ablation_runC_hutchinson", "D": "ablation_runD_detach_logdet",
               "E": "ablation_runE_vres_l2"}
    run_name = run_map[args.run]
    ckpt_dir = f"/home/tw559/steering_pi/KL_ReinFlow/log/gym/finetune/{run_name}/checkpoint"
    device = torch.device(args.device)
    iters = [int(x) for x in args.iters.split(",")]

    model = build_model(device)
    model.eval()

    # Fixed noise across all checkpoints
    torch.manual_seed(0)
    z_batch = torch.randn(args.n_obs, model.horizon_steps, model.action_dim, device=device)

    for it in iters:
        ckpt_path = f"{ckpt_dir}/state_{it}.pt"
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"  CHECKPOINT iter {it}")
        print(f"{'='*70}")

        data = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(data["model"])
        model.eval()

        # Collect real observations by rolling out this checkpoint's policy
        print(f"  Collecting {args.n_obs} observations from env rollout...")
        obs_np = collect_obs_from_env(model, device, n_obs=args.n_obs)
        obs_batch = torch.from_numpy(obs_np).float().to(device)
        print(f"  Collected. obs range: [{obs_batch.min():.3f}, {obs_batch.max():.3f}]")

        decompose_logpbase(model, obs_batch, z_batch, device, label=f"iter {it}")


if __name__ == "__main__":
    main()
