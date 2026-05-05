"""
Diagnose KL divergence root cause: test whether fine-tuned trajectory
drifts into OOD regions for the base model, causing ill-conditioned Jacobians.

Hypothesis: as v_res grows, the combined trajectory a^k_theta deviates from
the base trajectory a^k_base. At later ODE steps, the base model's Jacobian
evaluated at the combined trajectory's points becomes ill-conditioned
(I + J_base * dt near-singular), which is what causes slogdet backward to NaN.

Diagnostics measured per checkpoint:
  1. Trajectory divergence per step: ||a^k_theta - a^k_base||
  2. cond(I + J_base(a^k_theta) * dt) — Jacobian conditioning at theta's trajectory
  3. cond(I + J_base(a^k_base) * dt) — Jacobian conditioning at base's trajectory (control)
  4. sigma_min(I + J_base(a^k_theta) * dt) — proximity to singular
  5. log_p_base(a^{K-1}_theta) — base density at combined endpoint
  6. ||v_res(a^k, t_k)|| per step

Usage:
    python script/diagnose_kl_drift.py [--run RUN_NAME] [--iters 0,5000,20000,...]
"""
import argparse
import sys
import os
import json
import copy

import torch
import torch.nn as nn
import numpy as np
from torch.func import jacrev, vmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.flow.mlp_flow import FlowMLP
from model.flow.ft_sac.sac_residual_flow import SACResidualFlow, SigmaHead
from model.common.critic import CriticObsAct


def build_model(device, base_policy_path):
    """Build a SACResidualFlow model with the same architecture as the ablation runs."""
    obs_dim = 11
    action_dim = 3
    horizon_steps = 4
    cond_dim = obs_dim  # cond_steps=1

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
        actor_policy_path=base_policy_path,
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


@torch.no_grad()
def diagnose_checkpoint(model, obs_batch, z_batch, device):
    """Run diagnostics on a single checkpoint.

    Args:
        model: SACResidualFlow with weights loaded
        obs_batch: (N, obs_dim) observations
        z_batch: (N, horizon_steps, action_dim) fixed initial noise

    Returns:
        dict of diagnostic metrics
    """
    B = z_batch.shape[0]
    K = model.inference_steps
    dt = 1.0 / K
    D = model.act_dim_total
    I_D = torch.eye(D, device=device).unsqueeze(0)  # (1, D, D)
    cond = {"state": obs_batch}

    # ---------- Run combined ODE (v_base + v_res) ----------
    a_theta = z_batch.clone()
    theta_traj = [a_theta.clone()]
    theta_times = []

    for k in range(K - 1):
        t = torch.full((B,), k * dt, device=device)
        theta_times.append(t)
        v = model._combined_velocity(a_theta, t, cond)
        a_theta = a_theta + v * dt
        theta_traj.append(a_theta.clone())

    # ---------- Run base ODE (v_base only) ----------
    a_base = z_batch.clone()
    base_traj = [a_base.clone()]

    for k in range(K - 1):
        t = torch.full((B,), k * dt, device=device)
        v = model.v_base(a_base, t, cond)
        a_base = a_base + v * dt
        base_traj.append(a_base.clone())

    # ---------- Compute diagnostics per step ----------
    results = {
        "traj_divergence": [],       # ||a^k_theta - a^k_base|| per step
        "vres_norm": [],             # ||v_res(a^k_theta, t_k)|| per step
        "cond_at_theta": [],         # cond(M_base) at theta's trajectory
        "cond_at_base": [],          # cond(M_base) at base's trajectory (control)
        "sigma_min_at_theta": [],    # sigma_min(M_base) at theta's trajectory
        "sigma_min_at_base": [],     # sigma_min(M_base) at base's trajectory
        "logdet_at_theta": [],       # log|det(M_base)| at theta's trajectory
        "logdet_at_base": [],        # log|det(M_base)| at base's trajectory
    }

    for k in range(K - 1):
        t = theta_times[k]
        a_th_k = theta_traj[k]   # (B, Ta, Da)
        a_ba_k = base_traj[k]    # (B, Ta, Da)

        # 1. Trajectory divergence
        div = (a_th_k - a_ba_k).view(B, -1).norm(dim=-1)  # (B,)
        results["traj_divergence"].append({
            "mean": div.mean().item(),
            "max": div.max().item(),
            "std": div.std().item(),
        })

        # 2. ||v_res|| at theta's trajectory
        v_res_k = model.v_res(a_th_k, t, cond)
        vres_norm = v_res_k.view(B, -1).norm(dim=-1)
        results["vres_norm"].append({
            "mean": vres_norm.mean().item(),
            "max": vres_norm.max().item(),
        })

        # 3. Base Jacobian at theta's trajectory points
        J_theta = vmap(jacrev(model._per_sample_base_velocity_flat, argnums=0))(
            a_th_k.view(B, D), t, cond["state"]
        )
        M_theta = I_D + J_theta * dt
        sv_theta = torch.linalg.svdvals(M_theta)  # (B, D)
        cond_theta = sv_theta[:, 0] / sv_theta[:, -1].clamp(min=1e-30)
        smin_theta = sv_theta[:, -1]
        _, logdet_theta = torch.linalg.slogdet(M_theta)

        results["cond_at_theta"].append({
            "mean": cond_theta.mean().item(),
            "max": cond_theta.max().item(),
            "median": cond_theta.median().item(),
        })
        results["sigma_min_at_theta"].append({
            "mean": smin_theta.mean().item(),
            "min": smin_theta.min().item(),
            "median": smin_theta.median().item(),
        })
        results["logdet_at_theta"].append({
            "mean": logdet_theta.mean().item(),
            "std": logdet_theta.std().item(),
        })

        # 4. Base Jacobian at base's own trajectory (control)
        J_base = vmap(jacrev(model._per_sample_base_velocity_flat, argnums=0))(
            a_ba_k.view(B, D), t, cond["state"]
        )
        M_base = I_D + J_base * dt
        sv_base = torch.linalg.svdvals(M_base)
        cond_base = sv_base[:, 0] / sv_base[:, -1].clamp(min=1e-30)
        smin_base = sv_base[:, -1]
        _, logdet_base = torch.linalg.slogdet(M_base)

        results["cond_at_base"].append({
            "mean": cond_base.mean().item(),
            "max": cond_base.max().item(),
            "median": cond_base.median().item(),
        })
        results["sigma_min_at_base"].append({
            "mean": smin_base.mean().item(),
            "min": smin_base.min().item(),
            "median": smin_base.median().item(),
        })
        results["logdet_at_base"].append({
            "mean": logdet_base.mean().item(),
            "std": logdet_base.std().item(),
        })

    # ---------- 5. Base log-density at combined endpoint ----------
    a_Km1_theta = theta_traj[-1]  # a^{K-1} from combined ODE
    a_Km1_base = base_traj[-1]    # a^{K-1} from base ODE

    # log p_base(a^{K-1}_theta) via backward_base_logprob
    try:
        log_p_base_at_theta = model.backward_base_logprob(a_Km1_theta, cond)
        results["log_p_base_at_theta"] = {
            "mean": log_p_base_at_theta.mean().item(),
            "std": log_p_base_at_theta.std().item(),
            "min": log_p_base_at_theta.min().item(),
        }
    except Exception as e:
        results["log_p_base_at_theta"] = {"error": str(e)}

    # log p_base(a^{K-1}_base) as control
    try:
        log_p_base_at_base = model.backward_base_logprob(a_Km1_base, cond)
        results["log_p_base_at_base"] = {
            "mean": log_p_base_at_base.mean().item(),
            "std": log_p_base_at_base.std().item(),
            "min": log_p_base_at_base.min().item(),
        }
    except Exception as e:
        results["log_p_base_at_base"] = {"error": str(e)}

    # ---------- 6. Overall trajectory stats ----------
    a_Km1_div = (a_Km1_theta - a_Km1_base).view(B, -1).norm(dim=-1)
    results["endpoint_divergence"] = {
        "mean": a_Km1_div.mean().item(),
        "max": a_Km1_div.max().item(),
    }
    results["endpoint_theta_norm"] = a_Km1_theta.view(B, -1).norm(dim=-1).mean().item()
    results["endpoint_base_norm"] = a_Km1_base.view(B, -1).norm(dim=-1).mean().item()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="A",
                        help="Run to diagnose: A, B, C, D, E")
    parser.add_argument("--iters", type=str, default="0,5000,10000,20000,50000,100000,150000,199999",
                        help="Comma-separated checkpoint iterations")
    parser.add_argument("--n_samples", type=int, default=256,
                        help="Number of samples for diagnostics")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_map = {
        "A": "ablation_runA_baseline",
        "B": "ablation_runB_reward_penalty",
        "C": "ablation_runC_hutchinson",
        "D": "ablation_runD_detach_logdet",
        "E": "ablation_runE_vres_l2",
    }
    run_name = run_map[args.run]
    ckpt_dir = f"/home/tw559/steering_pi/KL_ReinFlow/log/gym/finetune/{run_name}/checkpoint"
    base_policy_path = "/home/tw559/steering_pi/KL_ReinFlow/log/gym/pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_D4RL_42/state_40.pt"

    iters = [int(x) for x in args.iters.split(",")]
    device = torch.device(args.device)

    # Build model
    print(f"Building model for run {args.run} ({run_name})...")
    model = build_model(device, base_policy_path)
    model.eval()

    # Fixed noise and observations (same across checkpoints for fair comparison)
    torch.manual_seed(0)
    z_batch = torch.randn(args.n_samples, model.horizon_steps, model.action_dim, device=device)

    # Use random observations (we don't need real env states for this diagnostic —
    # the hypothesis is about action-space drift, not state-dependent)
    # But normalize to roughly match training distribution
    norm_path = "/home/tw559/steering_pi/KL_ReinFlow/hf_cache/data-offline/gym/hopper-medium-v2/normalization.npz"
    if os.path.exists(norm_path):
        norm = np.load(norm_path)
        obs_min = torch.from_numpy(norm["obs_min"]).float().to(device)
        obs_max = torch.from_numpy(norm["obs_max"]).float().to(device)
        obs_mid = (obs_min + obs_max) / 2
        obs_range = (obs_max - obs_min).clamp(min=1e-6) / 2
        obs_batch = obs_mid.unsqueeze(0) + obs_range.unsqueeze(0) * torch.rand(args.n_samples, model.obs_dim, device=device) * 2 - obs_range.unsqueeze(0)
    else:
        print("Warning: normalization file not found, using random obs")
        obs_batch = torch.randn(args.n_samples, model.obs_dim, device=device)

    all_results = {}
    for it in iters:
        ckpt_path = f"{ckpt_dir}/state_{it}.pt"
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint {ckpt_path} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Checkpoint: iter {it}")
        print(f"{'='*60}")

        # Load checkpoint
        data = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(data["model"])
        model.eval()

        results = diagnose_checkpoint(model, obs_batch, z_batch, device)
        all_results[it] = results

        # Print summary
        K_minus_1 = model.inference_steps - 1
        print(f"\n  Per-step diagnostics (K-1 = {K_minus_1} steps):")
        print(f"  {'Step':>4} | {'traj_div':>10} | {'||v_res||':>10} | {'cond@theta':>12} | {'cond@base':>12} | {'smin@theta':>12} | {'smin@base':>12}")
        print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
        for k in range(K_minus_1):
            td = results["traj_divergence"][k]
            vn = results["vres_norm"][k]
            ct = results["cond_at_theta"][k]
            cb = results["cond_at_base"][k]
            st = results["sigma_min_at_theta"][k]
            sb = results["sigma_min_at_base"][k]
            print(f"  {k:>4} | {td['mean']:>10.4f} | {vn['mean']:>10.4f} | {ct['mean']:>12.2f} | {cb['mean']:>12.2f} | {st['mean']:>12.6f} | {sb['mean']:>12.6f}")

        print(f"\n  Endpoint divergence: mean={results['endpoint_divergence']['mean']:.4f}  max={results['endpoint_divergence']['max']:.4f}")
        print(f"  Endpoint norms: theta={results['endpoint_theta_norm']:.4f}  base={results['endpoint_base_norm']:.4f}")

        lpt = results.get("log_p_base_at_theta", {})
        lpb = results.get("log_p_base_at_base", {})
        if "error" not in lpt:
            print(f"  log_p_base at theta endpoint: mean={lpt['mean']:.2f}  min={lpt['min']:.2f}  std={lpt['std']:.2f}")
        else:
            print(f"  log_p_base at theta endpoint: ERROR — {lpt['error']}")
        if "error" not in lpb:
            print(f"  log_p_base at base endpoint:  mean={lpb['mean']:.2f}  min={lpb['min']:.2f}  std={lpb['std']:.2f}")

        # Worst-case condition numbers across all steps
        worst_cond_theta = max(r["max"] for r in results["cond_at_theta"])
        worst_smin_theta = min(r["min"] for r in results["sigma_min_at_theta"])
        worst_cond_base = max(r["max"] for r in results["cond_at_base"])
        worst_smin_base = min(r["min"] for r in results["sigma_min_at_base"])
        print(f"\n  Worst-case across steps:")
        print(f"    cond(M_base) at theta traj: {worst_cond_theta:.2f}")
        print(f"    cond(M_base) at base traj:  {worst_cond_base:.2f}")
        print(f"    sigma_min at theta traj:    {worst_smin_theta:.8f}")
        print(f"    sigma_min at base traj:     {worst_smin_base:.8f}")

    # Save full results
    out_path = f"/home/tw559/steering_pi/KL_ReinFlow/log/kl_drift_diagnostics_run{args.run}.json"
    # Convert to serializable
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nFull results saved to {out_path}")

    # Print trend summary
    print(f"\n{'='*60}")
    print(f"  TREND SUMMARY — Run {args.run}")
    print(f"{'='*60}")
    print(f"  {'Iter':>8} | {'endpoint_div':>12} | {'||v_res|| k=2':>14} | {'cond@theta k=2':>15} | {'smin@theta k=2':>15} | {'log_p_base@theta':>17}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*14}-+-{'-'*15}-+-{'-'*15}-+-{'-'*17}")
    for it in sorted(all_results.keys()):
        r = all_results[it]
        last_k = len(r["traj_divergence"]) - 1
        ed = r["endpoint_divergence"]["mean"]
        vn = r["vres_norm"][last_k]["mean"]
        ct = r["cond_at_theta"][last_k]["mean"]
        st = r["sigma_min_at_theta"][last_k]["mean"]
        lp = r.get("log_p_base_at_theta", {}).get("mean", float("nan"))
        print(f"  {it:>8} | {ed:>12.4f} | {vn:>14.4f} | {ct:>15.2f} | {st:>15.8f} | {lp:>17.2f}")


if __name__ == "__main__":
    main()
