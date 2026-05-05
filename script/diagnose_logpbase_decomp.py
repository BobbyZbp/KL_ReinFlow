"""
Decompose log_p_base to understand why it's so extreme (-100k nats).

log_p_base(a^{K-1}) = log_p_0(a^0_base) - sum_k log|det(I + J_base * dt)|

Which component is responsible?
  1. log_p_0(a^0_base) = -0.5*||a^0_base||^2 - D/2*log(2*pi)
     If the recovered a^0_base is far from origin, this is very negative.
  2. sum_logdet — accumulated volume change along the recovered trajectory.

Also checks fixed-point inversion accuracy:
  - Recover a^0_base from a^{K-1} via FP backward
  - Run base ODE forward from a^0_base
  - Compare reconstructed a^{K-1}_recon to original a^{K-1}
  - Round-trip error = ||a^{K-1}_recon - a^{K-1}||

Runs on both a^{K-1}_theta (combined endpoint) and a^{K-1}_base (base endpoint, control).
"""
import sys, os, math
import torch
import numpy as np
from torch.func import jacrev, vmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.flow.mlp_flow import FlowMLP
from model.flow.ft_sac.sac_residual_flow import SACResidualFlow, SigmaHead
from model.common.critic import CriticObsAct


def build_model(device):
    obs_dim, action_dim, horizon_steps, cond_dim = 11, 3, 4, 11
    base_policy_path = "/home/tw559/steering_pi/KL_ReinFlow/log/gym/pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_D4RL_42/state_40.pt"
    base_policy = FlowMLP(horizon_steps=horizon_steps, action_dim=action_dim, cond_dim=cond_dim,
        time_dim=16, mlp_dims=[512, 512, 512], activation_type="ReLU",
        out_activation_type="Identity", use_layernorm=False, residual_style=True)
    residual_policy = FlowMLP(horizon_steps=horizon_steps, action_dim=action_dim, cond_dim=cond_dim,
        time_dim=16, mlp_dims=[128, 128], activation_type="ReLU",
        out_activation_type="Identity", use_layernorm=False, residual_style=False)
    critic = CriticObsAct(cond_dim=cond_dim, mlp_dims=[256, 256, 256], action_dim=action_dim,
        action_steps=horizon_steps, activation_type="Mish", use_layernorm=True,
        residual_tyle=False, double_q=True)
    sigma_head = SigmaHead(cond_dim=cond_dim, action_dim=action_dim, horizon_steps=horizon_steps,
        hidden_dims=[64, 64], sigma_min=0.05, sigma_max=0.15)
    model = SACResidualFlow(device=device, base_policy=base_policy, residual_policy=residual_policy,
        critic=critic, sigma_head=sigma_head, actor_policy_path=base_policy_path,
        act_dim=action_dim, horizon_steps=horizon_steps, act_min=-1, act_max=1,
        obs_dim=obs_dim, cond_steps=1, inference_steps=4, backward_fp_iters=30,
        alpha=0.1, kl_weight=0.05, jac_weight=0.0, sigma_entropy_weight=0.1,
        zero_init_residual=True, kl_mode="none")
    return model


@torch.no_grad()
def forward_base_ode(model, a0, cond):
    """Run base ODE forward from a0, return trajectory and final point."""
    B = a0.shape[0]
    K = model.inference_steps
    dt = 1.0 / K
    a = a0.clone()
    traj = [a.clone()]
    for k in range(K - 1):
        t = torch.full((B,), k * dt, device=a.device)
        v = model.v_base(a, t, cond)
        a = a + v * dt
        traj.append(a.clone())
    return traj


@torch.no_grad()
def decompose_logpbase(model, a_Km1, cond, label=""):
    """Decompose log_p_base(a^{K-1}) into log_p_0 and sum_logdet, and check FP accuracy."""
    B = a_Km1.shape[0]
    device = a_Km1.device
    K = model.inference_steps
    dt = 1.0 / K
    D = model.act_dim_total
    I_D = torch.eye(D, device=device).unsqueeze(0)

    # Step 1: Recover base trajectory via FP backward
    traj_a_nograd, traj_t = model._backward_base_recover_trajectory(a_Km1, cond)
    a0_recovered = traj_a_nograd[0]  # recovered initial noise

    # Step 2: Check round-trip accuracy — forward base ODE from recovered a^0
    fwd_traj = forward_base_ode(model, a0_recovered, cond)
    a_Km1_recon = fwd_traj[-1]
    roundtrip_err = (a_Km1_recon - a_Km1).view(B, -1).norm(dim=-1)

    # Step 3: Decompose log_p_base
    # 3a: log_p_0(a^0_base)
    a0_norm = a0_recovered.view(B, -1).norm(dim=-1)
    log_p0 = -0.5 * (a0_recovered ** 2).sum(dim=(-2, -1)) - 0.5 * D * math.log(2 * math.pi)

    # 3b: sum_logdet along recovered trajectory
    sum_logdet = torch.zeros(B, device=device)
    per_step_logdet = []
    for k in range(K - 1):
        a_k = traj_a_nograd[k]
        t_k = traj_t[k]
        J = vmap(jacrev(model._per_sample_base_velocity_flat, argnums=0))(
            a_k.view(B, D), t_k, cond["state"]
        )
        M = I_D + J * dt
        sign, logabsdet = torch.linalg.slogdet(M)
        sum_logdet = sum_logdet + logabsdet
        per_step_logdet.append({
            "mean": logabsdet.mean().item(),
            "std": logabsdet.std().item(),
            "min": logabsdet.min().item(),
            "max": logabsdet.max().item(),
            "neg_sign_count": (sign < 0).sum().item(),
        })

    log_p_base = log_p0 - sum_logdet

    # Also check per-step FP accuracy: forward from a^k to a^{k+1} and compare
    per_step_fp_err = []
    for k in range(K - 2):
        a_k = traj_a_nograd[k]
        t_k = traj_t[k]
        v = model.v_base(a_k, t_k, cond)
        a_kp1_fwd = a_k + v * dt
        a_kp1_recovered = traj_a_nograd[k + 1]
        step_err = (a_kp1_fwd - a_kp1_recovered).view(B, -1).norm(dim=-1)
        per_step_fp_err.append({
            "mean": step_err.mean().item(),
            "max": step_err.max().item(),
        })
    # Last step: a^{K-2} → a^{K-1} (should match input)
    a_last_recovered = traj_a_nograd[-1]
    t_last = traj_t[-1]
    v_last = model.v_base(a_last_recovered, t_last, cond)
    a_Km1_from_last = a_last_recovered + v_last * dt
    last_step_err = (a_Km1_from_last - a_Km1).view(B, -1).norm(dim=-1)
    per_step_fp_err.append({
        "mean": last_step_err.mean().item(),
        "max": last_step_err.max().item(),
    })

    print(f"\n  === {label} ===")
    print(f"  Round-trip error ||a^{{K-1}}_recon - a^{{K-1}}||:")
    print(f"    mean={roundtrip_err.mean().item():.6f}  max={roundtrip_err.max().item():.6f}  median={roundtrip_err.median().item():.6f}")
    print(f"  Recovered a^0 norm: mean={a0_norm.mean().item():.4f}  max={a0_norm.max().item():.4f}  median={a0_norm.median().item():.4f}")
    print(f"  log_p_0(a^0_base):  mean={log_p0.mean().item():.2f}  min={log_p0.min().item():.2f}  std={log_p0.std().item():.2f}")
    print(f"  sum_logdet:         mean={sum_logdet.mean().item():.2f}  min={sum_logdet.min().item():.2f}  std={sum_logdet.std().item():.2f}")
    print(f"  log_p_base:         mean={log_p_base.mean().item():.2f}  min={log_p_base.min().item():.2f}  std={log_p_base.std().item():.2f}")
    print(f"  Contribution: log_p_0 accounts for {abs(log_p0.mean().item()) / (abs(log_p0.mean().item()) + abs(sum_logdet.mean().item()) + 1e-30) * 100:.1f}% of |log_p_base|")

    print(f"\n  Per-step logdet:")
    for k, ld in enumerate(per_step_logdet):
        print(f"    step {k}: mean={ld['mean']:.4f}  std={ld['std']:.4f}  min={ld['min']:.4f}  max={ld['max']:.4f}  neg_sign={ld['neg_sign_count']}")

    print(f"\n  Per-step FP consistency (||a^{{k+1}}_fwd - a^{{k+1}}_recovered||):")
    for k, fe in enumerate(per_step_fp_err):
        print(f"    step {k}→{k+1}: mean={fe['mean']:.6f}  max={fe['max']:.6f}")

    return {
        "roundtrip_err_mean": roundtrip_err.mean().item(),
        "roundtrip_err_max": roundtrip_err.max().item(),
        "a0_norm_mean": a0_norm.mean().item(),
        "a0_norm_max": a0_norm.max().item(),
        "log_p0_mean": log_p0.mean().item(),
        "sum_logdet_mean": sum_logdet.mean().item(),
        "log_p_base_mean": log_p_base.mean().item(),
        "per_step_logdet": per_step_logdet,
        "per_step_fp_err": per_step_fp_err,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="A")
    parser.add_argument("--iters", type=str, default="0,5000,20000,50000,100000,199999")
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_map = {"A": "ablation_runA_baseline", "B": "ablation_runB_reward_penalty",
               "C": "ablation_runC_hutchinson", "D": "ablation_runD_detach_logdet",
               "E": "ablation_runE_vres_l2"}
    run_name = run_map[args.run]
    ckpt_dir = f"/home/tw559/steering_pi/KL_ReinFlow/log/gym/finetune/{run_name}/checkpoint"
    device = torch.device(args.device)

    model = build_model(device)
    model.eval()

    torch.manual_seed(0)
    z_batch = torch.randn(args.n_samples, model.horizon_steps, model.action_dim, device=device)

    norm_path = "/home/tw559/steering_pi/KL_ReinFlow/hf_cache/data-offline/gym/hopper-medium-v2/normalization.npz"
    norm = np.load(norm_path)
    obs_min = torch.from_numpy(norm["obs_min"]).float().to(device)
    obs_max = torch.from_numpy(norm["obs_max"]).float().to(device)
    obs_mid = (obs_min + obs_max) / 2
    obs_range = (obs_max - obs_min).clamp(min=1e-6) / 2
    obs_batch = obs_mid.unsqueeze(0) + obs_range.unsqueeze(0) * (torch.rand(args.n_samples, model.obs_dim, device=device) * 2 - 1)
    cond = {"state": obs_batch}

    iters = [int(x) for x in args.iters.split(",")]

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

        K = model.inference_steps
        dt = 1.0 / K
        B = z_batch.shape[0]

        # Run combined ODE
        a_theta = z_batch.clone()
        for k in range(K - 1):
            t = torch.full((B,), k * dt, device=device)
            v = model._combined_velocity(a_theta, t, cond)
            a_theta = a_theta + v * dt
        a_Km1_theta = a_theta

        # Run base ODE
        a_base = z_batch.clone()
        for k in range(K - 1):
            t = torch.full((B,), k * dt, device=device)
            v = model.v_base(a_base, t, cond)
            a_base = a_base + v * dt
        a_Km1_base = a_base

        div = (a_Km1_theta - a_Km1_base).view(B, -1).norm(dim=-1)
        print(f"  Endpoint divergence: mean={div.mean().item():.4f}  max={div.max().item():.4f}")

        # Decompose log_p_base at theta's endpoint
        decompose_logpbase(model, a_Km1_theta, cond, label="log_p_base at THETA endpoint")

        # Decompose log_p_base at base's own endpoint (control)
        decompose_logpbase(model, a_Km1_base, cond, label="log_p_base at BASE endpoint (control)")


if __name__ == "__main__":
    main()
