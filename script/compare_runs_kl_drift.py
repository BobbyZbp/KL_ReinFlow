"""
Diagnostic #2 & #3: Compare KL divergence and trajectory drift between runs.

For matched checkpoints across runs (e.g., A2=baseline vs B=reward_penalty):
  - Compute KL(p_theta || p_base) via per-step method (stable, no endpoint inversion)
  - Measure ||v_res|| growth (proxy for policy deviation magnitude)
  - Measure trajectory divergence ||a^k_theta - a^k_base|| per step
  - Compute per-step KL value (logdet difference at inverted points)

Goal: show whether the KL reward penalty (run B) actually maintains lower KL
and slower drift compared to baseline (run A2), even if reward is similar.

Usage:
    conda run -n reinflow python script/compare_runs_kl_drift.py --device cpu
    conda run -n reinflow python script/compare_runs_kl_drift.py --device cuda:0
"""
import argparse
import sys
import os
import json

import torch
import numpy as np
from torch.func import jacrev, vmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.flow.mlp_flow import FlowMLP
from model.flow.ft_sac.sac_residual_flow import SACResidualFlow, SigmaHead
from model.common.critic import CriticObsAct


BASE_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "log/gym/pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_D4RL_42/state_40.pt"
)

NORM_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hf_cache/data-offline/gym/hopper-medium-v2/normalization.npz"
)

RUNS = {
    "A2": "perstep_runA2_baseline",
    "B": "ablation_runB_reward_penalty",
}


def build_model(device):
    obs_dim = 11
    action_dim = 3
    horizon_steps = 4
    cond_dim = obs_dim

    base_policy = FlowMLP(
        horizon_steps=horizon_steps, action_dim=action_dim, cond_dim=cond_dim,
        time_dim=16, mlp_dims=[512, 512, 512], activation_type="ReLU",
        out_activation_type="Identity", use_layernorm=False, residual_style=True,
    )
    residual_policy = FlowMLP(
        horizon_steps=horizon_steps, action_dim=action_dim, cond_dim=cond_dim,
        time_dim=16, mlp_dims=[128, 128], activation_type="ReLU",
        out_activation_type="Identity", use_layernorm=False, residual_style=False,
    )
    critic = CriticObsAct(
        cond_dim=cond_dim, mlp_dims=[256, 256, 256], action_dim=action_dim,
        action_steps=horizon_steps, activation_type="Mish",
        use_layernorm=True, residual_tyle=False, double_q=True,
    )
    sigma_head = SigmaHead(
        cond_dim=cond_dim, action_dim=action_dim, horizon_steps=horizon_steps,
        hidden_dims=[64, 64], sigma_min=0.05, sigma_max=0.15,
    )

    model = SACResidualFlow(
        device=device,
        base_policy=base_policy,
        residual_policy=residual_policy,
        critic=critic,
        sigma_head=sigma_head,
        actor_policy_path=BASE_POLICY_PATH,
        act_dim=action_dim,
        horizon_steps=horizon_steps,
        act_min=-1, act_max=1,
        obs_dim=obs_dim,
        cond_steps=1,
        inference_steps=4,
        denoised_clip_value=1.0,
        randn_clip_value=3.0,
        backward_fp_iters=30,
        alpha=0.1,
        kl_weight=0.05,
        jac_weight=0.0,
        sigma_entropy_weight=0.1,
        target_ema_rate=0.005,
        zero_init_residual=True,
        kl_mode="none",
        vres_l2_weight=0.0,
        hutchinson_samples=1,
    )
    return model


def collect_real_obs(model, device, n_obs=256):
    """Roll out the base policy in Hopper-v2 to collect real observations."""
    import gym
    import d4rl.gym_mujoco

    norm = np.load(NORM_PATH)
    obs_min_np = norm["obs_min"]
    obs_max_np = norm["obs_max"]

    env = gym.make("hopper-medium-v2")
    collected = []
    obs = env.reset()

    while len(collected) < n_obs:
        obs_norm = -1.0 + 2.0 * (obs - obs_min_np) / (obs_max_np - obs_min_np + 1e-8)
        obs_norm = np.clip(obs_norm, -1.0, 1.0)
        collected.append(obs_norm.copy())

        with torch.no_grad():
            obs_t = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
            cond = {"state": obs_t}
            a = model.sample_action(cond, deterministic=False)
            action = a[0, :1].cpu().numpy().flatten()[:3]
            action = np.clip(action, -1.0, 1.0)

        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    env.close()
    obs_array = np.array(collected[:n_obs])
    return torch.from_numpy(obs_array).float().to(device)


@torch.no_grad()
def compute_metrics(model, obs_batch, z_batch, device, fp_iters=10):
    """Compute KL, drift, and v_res magnitude for a loaded checkpoint.

    Uses per-step KL (exact slogdet) — stable computation that doesn't require
    endpoint backward inversion.
    """
    B = z_batch.shape[0]
    K = model.inference_steps
    dt = 1.0 / K
    D = model.act_dim_total
    I_D = torch.eye(D, device=device).unsqueeze(0)
    cond = {"state": obs_batch}

    # ---- Combined ODE (v_base + v_res) ----
    a_theta = z_batch.clone()
    theta_traj = [a_theta.clone()]
    for k in range(K - 1):
        t = torch.full((B,), k * dt, device=device)
        v = model._combined_velocity(a_theta, t, cond)
        a_theta = a_theta + v * dt
        theta_traj.append(a_theta.clone())

    # ---- Base ODE (v_base only, same initial noise) ----
    a_base = z_batch.clone()
    base_traj = [a_base.clone()]
    for k in range(K - 1):
        t = torch.full((B,), k * dt, device=device)
        v = model.v_base(a_base, t, cond)
        a_base = a_base + v * dt
        base_traj.append(a_base.clone())

    # ---- Per-step metrics ----
    total_kl = torch.zeros(B, device=device)
    vres_norms = []
    traj_divs = []

    for k in range(K - 1):
        t_k = torch.full((B,), k * dt, device=device)
        a_k = theta_traj[k]
        a_next = theta_traj[k + 1]

        # v_res norm at this step
        v_res_k = model.v_res(a_k, t_k, cond)
        vres_norm_k = v_res_k.view(B, -1).norm(dim=-1).mean().item()
        vres_norms.append(vres_norm_k)

        # Trajectory divergence at this step
        div_k = (a_k - base_traj[k]).view(B, -1).norm(dim=-1).mean().item()
        traj_divs.append(div_k)

        # Per-step KL: logdet(I + J_base(a_inv)*dt) - logdet(I + J_combined(a_k)*dt)
        # One-step FP inversion
        a_inv = a_next - model.v_base(a_next, t_k, cond) * dt
        for _ in range(fp_iters):
            a_inv = a_next - model.v_base(a_inv, t_k, cond) * dt

        # logdet at inverted point (base Jacobian)
        J_base_inv = vmap(jacrev(model._per_sample_base_velocity_flat, argnums=0))(
            a_inv.view(B, D), t_k, cond["state"]
        )
        M_base_inv = I_D + J_base_inv * dt
        _, logdet_base = torch.linalg.slogdet(M_base_inv)

        # logdet at combined trajectory point (combined Jacobian)
        J_comb = vmap(jacrev(model._per_sample_combined_velocity_flat, argnums=0))(
            a_k.view(B, D), t_k, cond["state"]
        )
        M_comb = I_D + J_comb * dt
        _, logdet_comb = torch.linalg.slogdet(M_comb)

        # KL_k = logdet_base(a_inv) - logdet_combined(a_k)
        kl_k = logdet_base - logdet_comb
        total_kl = total_kl + kl_k

    # Endpoint divergence
    endpoint_div = (theta_traj[-1] - base_traj[-1]).view(B, -1).norm(dim=-1).mean().item()

    return {
        "perstep_kl_mean": total_kl.mean().item(),
        "perstep_kl_std": total_kl.std().item(),
        "perstep_kl_max": total_kl.max().item(),
        "perstep_kl_min": total_kl.min().item(),
        "vres_norms": vres_norms,
        "vres_total": sum(vres_norms),
        "traj_divs": traj_divs,
        "endpoint_div": endpoint_div,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--fp_iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    log_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "log/gym/finetune"
    )

    # Find common checkpoints between runs
    common_iters = None
    for run_key, run_dir in RUNS.items():
        ckpt_dir = os.path.join(log_base, run_dir, "checkpoint")
        available = set(
            int(f.replace("state_", "").replace(".pt", ""))
            for f in os.listdir(ckpt_dir)
            if f.startswith("state_") and f != "latest.pt"
        )
        if common_iters is None:
            common_iters = available
        else:
            common_iters = common_iters & available

    common_iters = sorted(common_iters)
    # Subsample for speed
    target_iters = [0, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
    iters = [i for i in target_iters if i in common_iters]
    if not iters:
        iters = common_iters[:10]

    print(f"Comparing runs: {list(RUNS.keys())}")
    print(f"Common checkpoints to analyze: {iters}")
    print(f"Device: {device}")

    model = build_model(device)
    model.eval()

    # Fixed noise
    torch.manual_seed(42)
    z_batch = torch.randn(args.n_samples, model.horizon_steps, model.action_dim, device=device)

    # Collect real observations from Hopper env (base policy rollout)
    obs_batch = collect_real_obs(model, device, args.n_samples)

    # Collect metrics for each run
    all_metrics = {}
    for run_key, run_dir in RUNS.items():
        ckpt_dir = os.path.join(log_base, run_dir, "checkpoint")
        print(f"\n{'='*60}")
        print(f"  Run {run_key} ({run_dir})")
        print(f"{'='*60}")

        run_metrics = {}
        for it in iters:
            model_only_path = os.path.join(ckpt_dir, "model_only", f"state_{it}.pt")
            full_path = os.path.join(ckpt_dir, f"state_{it}.pt")
            if os.path.exists(model_only_path):
                ckpt_path = model_only_path
            elif os.path.exists(full_path):
                ckpt_path = full_path
            else:
                continue

            data = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(data["model"], strict=False)
            del data
            model.eval()

            metrics = compute_metrics(model, obs_batch, z_batch, device, args.fp_iters)
            run_metrics[it] = metrics
            print(f"  iter {it:>6}: KL={metrics['perstep_kl_mean']:>8.3f} | "
                  f"||v_res||={metrics['vres_total']:>7.3f} | "
                  f"endpoint_div={metrics['endpoint_div']:>7.3f}")

        all_metrics[run_key] = run_metrics

    # ---- Comparison table ----
    print(f"\n{'='*70}")
    print(f"  COMPARISON: A2 (no KL) vs B (KL reward penalty)")
    print(f"{'='*70}")
    print(f"  {'iter':>6} | {'KL_A2':>8} {'KL_B':>8} {'ratio':>6} | "
          f"{'drift_A2':>8} {'drift_B':>8} {'ratio':>6} | "
          f"{'vres_A2':>8} {'vres_B':>8} {'ratio':>6}")
    print(f"  {'-'*6}-+-{'-'*25}-+-{'-'*25}-+-{'-'*25}")

    for it in iters:
        mA = all_metrics.get("A2", {}).get(it)
        mB = all_metrics.get("B", {}).get(it)
        if mA is None or mB is None:
            continue

        kl_a, kl_b = mA["perstep_kl_mean"], mB["perstep_kl_mean"]
        dr_a, dr_b = mA["endpoint_div"], mB["endpoint_div"]
        vr_a, vr_b = mA["vres_total"], mB["vres_total"]

        kl_ratio = kl_b / kl_a if abs(kl_a) > 1e-6 else float("nan")
        dr_ratio = dr_b / dr_a if abs(dr_a) > 1e-6 else float("nan")
        vr_ratio = vr_b / vr_a if abs(vr_a) > 1e-6 else float("nan")

        print(f"  {it:>6} | {kl_a:>8.3f} {kl_b:>8.3f} {kl_ratio:>5.2f}x | "
              f"{dr_a:>8.3f} {dr_b:>8.3f} {dr_ratio:>5.2f}x | "
              f"{vr_a:>8.3f} {vr_b:>8.3f} {vr_ratio:>5.2f}x")

    # ---- Interpretation ----
    last_iter = max(it for it in iters
                    if it in all_metrics.get("A2", {}) and it in all_metrics.get("B", {}))
    mA_last = all_metrics["A2"][last_iter]
    mB_last = all_metrics["B"][last_iter]
    print(f"\n  At iter {last_iter}:")
    print(f"    KL(A2) = {mA_last['perstep_kl_mean']:.3f},  KL(B) = {mB_last['perstep_kl_mean']:.3f}")
    print(f"    drift(A2) = {mA_last['endpoint_div']:.3f},  drift(B) = {mB_last['endpoint_div']:.3f}")
    if abs(mA_last['perstep_kl_mean']) > abs(mB_last['perstep_kl_mean']):
        print(f"    => B maintains LOWER KL (reward penalty constrains drift)")
    else:
        print(f"    => B has HIGHER KL (reward penalty not effective at this weight)")

    # Save
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "log/compare_runs_kl_drift.json"
    )
    serializable = {}
    for rk, rm in all_metrics.items():
        serializable[rk] = {str(k): v for k, v in rm.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
