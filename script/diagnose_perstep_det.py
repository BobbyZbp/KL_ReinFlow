"""
Diagnostic #1: Per-step determinant analysis at FP-inverted points.

Tests the claim: "the base model cannot produce the fine-tuned model's endpoint
action, but it CAN produce each one-step transition along the fine-tuned trajectory."

At each ODE step k, the combined policy produces:
    a^{k+1}_theta = a^k + (v_base + v_res)(a^k, t_k) * dt

Per-step KL requires inverting ONE step of the base map:
    Find a^k_inv such that a^k_inv + v_base(a^k_inv, t_k) * dt = a^{k+1}_theta

This script measures:
  - det(I + J_base(a^k_inv) * dt) at the inverted point (must be > 0 for per-step KL)
  - det(I + J_base(a^{K-1}_theta) * dt) at the endpoint (known to go < 0)
  - FP convergence: ||a^k_inv + v_base(a^k_inv)*dt - a^{k+1}||
  - Round-trip error: ||forward_base(a^k_inv) - a^{k+1}_theta||

If per-step det stays positive while endpoint det goes negative, that's direct
evidence for why per-step KL is stable and endpoint KL is not.

Usage:
    conda run -n reinflow python script/diagnose_perstep_det.py --run A2 --device cpu
    conda run -n reinflow python script/diagnose_perstep_det.py --run B --device cuda:0
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


RUN_MAP = {
    "A2": "perstep_runA2_baseline",
    "B": "ablation_runB_reward_penalty",
    "5b": "perstep_run5b_exact_highkl",
    "7": "perstep_run7_exact_ema",
}

BASE_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "log/gym/pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_D4RL_42/state_40.pt"
)

NORM_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hf_cache/data-offline/gym/hopper-medium-v2/normalization.npz"
)


def build_model(device, base_policy_path):
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
def diagnose_perstep_det(model, obs_batch, z_batch, device, fp_iters=10):
    """Core diagnostic: compare det at per-step inverted points vs endpoint.

    Returns dict with per-step and endpoint determinant statistics.
    """
    B = z_batch.shape[0]
    K = model.inference_steps
    dt = 1.0 / K
    D = model.act_dim_total
    I_D = torch.eye(D, device=device).unsqueeze(0)
    cond = {"state": obs_batch}

    # ---- Run combined ODE to get theta trajectory ----
    a = z_batch.clone()
    theta_traj = [a.clone()]
    for k in range(K - 1):
        t = torch.full((B,), k * dt, device=device)
        v = model._combined_velocity(a, t, cond)
        a = a + v * dt
        theta_traj.append(a.clone())

    # ---- Per-step: invert each transition via one-step FP ----
    perstep_results = []
    for k in range(K - 1):
        t_k = torch.full((B,), k * dt, device=device)
        a_k = theta_traj[k]         # start of step k
        a_next = theta_traj[k + 1]  # end of step k (= a^{k+1}_theta)

        # One-step FP inversion: find a_inv s.t. a_inv + v_base(a_inv, t_k)*dt = a_next
        a_inv = a_next - model.v_base(a_next, t_k, cond) * dt  # explicit init
        for _ in range(fp_iters):
            a_inv = a_next - model.v_base(a_inv, t_k, cond) * dt

        # Measure FP convergence: residual of the fixed-point equation
        fp_residual = (a_inv + model.v_base(a_inv, t_k, cond) * dt - a_next)
        fp_err = fp_residual.view(B, -1).norm(dim=-1)

        # Jacobian of base velocity at the INVERTED point
        J_at_inv = vmap(jacrev(model._per_sample_base_velocity_flat, argnums=0))(
            a_inv.view(B, D), t_k, cond["state"]
        )
        M_at_inv = I_D + J_at_inv * dt
        sign_inv, logdet_inv = torch.linalg.slogdet(M_at_inv)
        det_inv = sign_inv * logdet_inv.exp()

        # Jacobian of base velocity at the COMBINED trajectory point a^k
        J_at_theta = vmap(jacrev(model._per_sample_base_velocity_flat, argnums=0))(
            a_k.view(B, D), t_k, cond["state"]
        )
        M_at_theta = I_D + J_at_theta * dt
        sign_theta, logdet_theta = torch.linalg.slogdet(M_at_theta)

        # Distance between a_inv and a_k (how far did the inversion move?)
        inv_shift = (a_inv - a_k).view(B, -1).norm(dim=-1)

        neg_sign_inv = (sign_inv < 0).sum().item()
        neg_sign_theta = (sign_theta < 0).sum().item()

        perstep_results.append({
            "step": k,
            "neg_sign_at_inv": neg_sign_inv,
            "neg_sign_at_theta_traj": neg_sign_theta,
            "det_at_inv_mean": det_inv.mean().item(),
            "det_at_inv_min": det_inv.min().item(),
            "logdet_at_inv_mean": logdet_inv.mean().item(),
            "logdet_at_inv_std": logdet_inv.std().item(),
            "sign_at_inv_all_pos": bool(neg_sign_inv == 0),
            "fp_err_mean": fp_err.mean().item(),
            "fp_err_max": fp_err.max().item(),
            "inv_shift_mean": inv_shift.mean().item(),
            "inv_shift_max": inv_shift.max().item(),
        })

    # ---- Endpoint: try backward inversion of full base ODE from a^{K-1}_theta ----
    a_Km1_theta = theta_traj[-1]  # endpoint of combined ODE

    # Full backward inversion (K-1 sequential steps)
    a_curr = a_Km1_theta.clone()
    endpoint_neg_signs = 0
    endpoint_logdets = []
    for k in range(K - 2, -1, -1):
        t_k = torch.full((B,), k * dt, device=device)
        # FP inversion
        a_prev = a_curr - model.v_base(a_curr, t_k, cond) * dt
        for _ in range(30):
            a_prev = a_curr - model.v_base(a_prev, t_k, cond) * dt

        # Jacobian at inverted point
        J = vmap(jacrev(model._per_sample_base_velocity_flat, argnums=0))(
            a_prev.view(B, D), t_k, cond["state"]
        )
        M = I_D + J * dt
        sign, logdet = torch.linalg.slogdet(M)
        endpoint_neg_signs += (sign < 0).sum().item()
        endpoint_logdets.append(logdet.mean().item())
        a_curr = a_prev

    # Also check: det(I + J_base * dt) evaluated directly AT the endpoint
    t_last_det = torch.full((B,), (K - 2) * dt, device=device)
    J_endpoint = vmap(jacrev(model._per_sample_base_velocity_flat, argnums=0))(
        a_Km1_theta.view(B, D), t_last_det, cond["state"]
    )
    M_endpoint = I_D + J_endpoint * dt
    sign_ep, logdet_ep = torch.linalg.slogdet(M_endpoint)
    neg_at_endpoint_direct = (sign_ep < 0).sum().item()

    endpoint_results = {
        "neg_signs_in_backward_inversion": endpoint_neg_signs,
        "neg_at_endpoint_direct": neg_at_endpoint_direct,
        "total_samples": B * (K - 1),
        "endpoint_logdets": endpoint_logdets,
    }

    # ---- Summary ----
    return {
        "perstep": perstep_results,
        "endpoint": endpoint_results,
        "endpoint_norm": a_Km1_theta.view(B, -1).norm(dim=-1).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="A2",
                        help="Run to diagnose: A2, B, 5b, 7")
    parser.add_argument("--iters", type=str, default=None,
                        help="Comma-separated iters (default: auto-detect common set)")
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--fp_iters", type=int, default=10,
                        help="FP iterations for per-step inversion")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_name = RUN_MAP[args.run]
    ckpt_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        f"log/gym/finetune/{run_name}/checkpoint"
    )

    if args.iters is None:
        available = sorted([
            int(f.replace("state_", "").replace(".pt", ""))
            for f in os.listdir(ckpt_dir)
            if f.startswith("state_") and f != "latest.pt"
        ])
        iters = [i for i in [0, 5000, 10000, 20000, 40000, 60000, 80000] if i in available]
        if not iters:
            iters = available[:8]
    else:
        iters = [int(x) for x in args.iters.split(",")]

    device = torch.device(args.device)
    print(f"Run: {args.run} ({run_name})")
    print(f"Checkpoints: {iters}")
    print(f"Device: {device}")
    print(f"FP iters for per-step inversion: {args.fp_iters}")

    model = build_model(device, BASE_POLICY_PATH)
    model.eval()

    # Fixed noise
    torch.manual_seed(42)
    z_batch = torch.randn(args.n_samples, model.horizon_steps, model.action_dim, device=device)

    # Collect REAL observations by rolling out the base policy in Hopper.
    # Synthetic observations produce det<0 even at iter 0 (OOD for the base model).
    obs_batch = collect_real_obs(model, device, args.n_samples)

    all_results = {}
    for it in iters:
        # Prefer model_only/ (small files) over full checkpoints (include replay buffer)
        model_only_path = os.path.join(ckpt_dir, "model_only", f"state_{it}.pt")
        full_path = os.path.join(ckpt_dir, f"state_{it}.pt")
        if os.path.exists(model_only_path):
            ckpt_path = model_only_path
        elif os.path.exists(full_path):
            ckpt_path = full_path
        else:
            print(f"  iter {it}: not found, skipping")
            continue

        data = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(data["model"], strict=False)
        del data
        model.eval()

        results = diagnose_perstep_det(model, obs_batch, z_batch, device, args.fp_iters)
        all_results[it] = results

        # Print summary for this checkpoint
        print(f"\n{'='*70}")
        print(f"  iter {it}")
        print(f"{'='*70}")
        print(f"  endpoint norm: {results['endpoint_norm']:.3f}")
        print(f"  endpoint backward inversion: {results['endpoint']['neg_signs_in_backward_inversion']}/{results['endpoint']['total_samples']} neg signs")
        print(f"  endpoint direct det<0: {results['endpoint']['neg_at_endpoint_direct']}/{args.n_samples}")
        print()
        print(f"  {'step':>4} | {'neg@inv':>8} | {'neg@traj':>9} | {'det_inv_min':>11} | {'fp_err_max':>10} | {'inv_shift':>10}")
        print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*9}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}")
        for ps in results["perstep"]:
            print(f"  {ps['step']:>4} | {ps['neg_sign_at_inv']:>5}/{args.n_samples:>0} | "
                  f"{ps['neg_sign_at_theta_traj']:>6}/{args.n_samples:>0} | "
                  f"{ps['det_at_inv_min']:>11.6f} | "
                  f"{ps['fp_err_max']:>10.2e} | "
                  f"{ps['inv_shift_mean']:>10.4f}")

    # Print cross-checkpoint trend
    print(f"\n{'='*70}")
    print(f"  TREND: Per-step det (all positive?) vs endpoint det (goes negative?)")
    print(f"{'='*70}")
    print(f"  {'iter':>8} | {'perstep neg total':>18} | {'endpoint neg':>12} | {'endpoint_direct':>15}")
    print(f"  {'-'*8}-+-{'-'*18}-+-{'-'*12}-+-{'-'*15}")
    for it in sorted(all_results.keys()):
        r = all_results[it]
        ps_neg = sum(ps["neg_sign_at_inv"] for ps in r["perstep"])
        ps_total = args.n_samples * (model.inference_steps - 1)
        ep_neg = r["endpoint"]["neg_signs_in_backward_inversion"]
        ep_direct = r["endpoint"]["neg_at_endpoint_direct"]
        print(f"  {it:>8} | {ps_neg:>6}/{ps_total:<6} ({100*ps_neg/ps_total:>5.1f}%) | "
              f"{ep_neg:>5}/{ps_total} | "
              f"{ep_direct:>6}/{args.n_samples}")

    # Save
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        f"log/diagnose_perstep_det_run{args.run}.json"
    )
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
