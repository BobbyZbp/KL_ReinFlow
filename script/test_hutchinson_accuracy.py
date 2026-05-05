"""
Test Hutchinson trace estimator accuracy vs exact Jacobian trace/logdet.

Tests three Jacobians separately:
  - J_combined: d(v_base + v_res)/da  (used in forward_with_logdet for log p_theta)
  - J_base:     d(v_base)/da          (used in backward_base_logprob for log p_base)
  - J_res:      d(v_res)/da           (the only part with trainable params)

Measures two independent error sources:
  1. Hutchinson sampling error: tr_hutch(J) vs tr(J)
  2. Taylor approximation error: tr(J)*dt vs logdet(I + J*dt)

Probes are batched into the batch dimension for GPU parallelism.

Uses real observations from D4RL Hopper and real ODE trajectory points from
trained checkpoints. Never uses synthetic/random inputs.

Usage:
  python script/test_hutchinson_accuracy.py [--run A] [--iters 0,100000,199999]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.func import jacrev, vmap, jvp

from model.flow.mlp_flow import FlowMLP
from model.common.critic import CriticObsAct
from model.flow.ft_sac.sac_residual_flow import SACResidualFlow, SigmaHead

BASE_POLICY_PATH = "/home/tw559/steering_pi/KL_ReinFlow/log/gym/pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_D4RL_42/state_40.pt"
NORM_PATH = "/home/tw559/steering_pi/KL_ReinFlow/hf_cache/data-offline/gym/hopper-medium-v2/normalization.npz"


def make_model(device):
    action_dim, horizon_steps, obs_dim, cond_steps = 3, 4, 11, 1
    inference_steps = 4
    cond_dim = obs_dim * cond_steps
    base = FlowMLP(
        horizon_steps=horizon_steps, action_dim=action_dim, cond_dim=cond_dim,
        time_dim=16, mlp_dims=[512, 512, 512], activation_type="ReLU",
        out_activation_type="Identity", use_layernorm=False, residual_style=True,
    )
    res = FlowMLP(
        horizon_steps=horizon_steps, action_dim=action_dim, cond_dim=cond_dim,
        time_dim=16, mlp_dims=[128, 128], activation_type="ReLU",
        out_activation_type="Identity", use_layernorm=False, residual_style=False,
    )
    critic = CriticObsAct(
        cond_dim=cond_dim, mlp_dims=[256, 256, 256], action_dim=action_dim,
        action_steps=horizon_steps, activation_type="Mish",
        use_layernorm=True, residual_tyle=False, double_q=True,
    )
    sigma = SigmaHead(
        cond_dim=cond_dim, action_dim=action_dim, horizon_steps=horizon_steps,
        hidden_dims=[64, 64], sigma_min=0.05, sigma_max=0.15,
    )
    model = SACResidualFlow(
        device=device, base_policy=base, residual_policy=res, critic=critic,
        sigma_head=sigma, actor_policy_path=BASE_POLICY_PATH,
        act_dim=action_dim, horizon_steps=horizon_steps,
        act_min=-1.0, act_max=1.0, obs_dim=obs_dim, cond_steps=cond_steps,
        inference_steps=inference_steps,
        backward_fp_iters=30, alpha=0.1, kl_weight=0.05, jac_weight=0.0,
        sigma_entropy_weight=0.1,
        zero_init_residual=True, kl_mode="none",
    ).to(device)
    return model


def load_real_obs(device, n_obs=1024):
    """Load observations from D4RL hopper-medium-v2 offline dataset (cached HDF5)."""
    import h5py

    ds_path = os.path.expanduser("~/.d4rl/datasets/hopper_medium-v2.hdf5")
    norm = np.load(NORM_PATH)
    obs_min, obs_max = norm["obs_min"], norm["obs_max"]

    with h5py.File(ds_path, "r") as f:
        raw_obs = f["observations"][:n_obs]

    obs_norm = -1.0 + 2.0 * (raw_obs - obs_min) / (obs_max - obs_min + 1e-8)
    obs_norm = np.clip(obs_norm, -1.0, 1.0)
    return torch.from_numpy(obs_norm).float().to(device)


def hutchinson_trace_batched(vel_fn, a_flat, t, cond_state, n_probes,
                             B, D, horizon_steps, action_dim, device):
    """Estimate tr(J) by expanding probes into the batch dimension.

    Tiles (a, t, cond) n_probes times along batch dim, draws n_probes*B
    Rademacher vectors, computes all JVPs in one call, then reshapes and
    averages over the probe dimension.

    Returns: (B,) trace estimate.
    """
    NB = n_probes * B
    a_exp = a_flat.repeat(n_probes, 1)                    # (NB, D)
    t_exp = t.repeat(n_probes)                            # (NB,)
    cond_exp = cond_state.repeat(n_probes, *([1] * (cond_state.dim() - 1)))  # (NB, ...)

    probes = torch.randint(0, 2, (NB, D), device=device).float() * 2 - 1

    def vel_batched(af):
        aa = af.view(NB, horizon_steps, action_dim)
        return vel_fn(aa, t_exp, {"state": cond_exp}).view(NB, D)

    _, Jv = jvp(vel_batched, (a_exp,), (probes,))
    vJv = (probes * Jv).sum(dim=-1)                       # (NB,)
    return vJv.view(n_probes, B).mean(dim=0)              # (B,)


@torch.no_grad()
def test_jacobian(model, device, vel_fn, jacrev_fn, jac_name,
                  obs_pool, B=64, n_trials=100):
    """Test Hutchinson accuracy for one specific Jacobian.

    Uses real observations from obs_pool and real ODE trajectory points
    (by running the combined ODE forward from Gaussian noise).
    """
    action_dim = model.action_dim
    horizon_steps = model.horizon_steps
    D = model.act_dim_total
    K = model.inference_steps
    dt = 1.0 / K
    I_D = torch.eye(D, device=device).unsqueeze(0)
    N_pool = obs_pool.shape[0]

    probe_counts = [1, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512]

    print(f"\n  --- Jacobian: {jac_name} ---")

    all_results_per_step = {k: [] for k in range(K - 1)}

    for trial in range(n_trials):
        idx = torch.randint(0, N_pool, (B,), device=device)
        obs = obs_pool[idx]
        cond = {"state": obs}
        a = torch.randn(B, horizon_steps, action_dim, device=device)

        for k in range(K - 1):
            t = torch.full((B,), k * dt, device=device)
            a_flat = a.view(B, D)

            # exact Jacobian
            J = vmap(jacrev(jacrev_fn, argnums=0))(
                a_flat, t, cond["state"]
            )
            exact_trace = torch.diagonal(J, dim1=-2, dim2=-1).sum(dim=-1)
            M = I_D + J * dt
            _, exact_logdet = torch.linalg.slogdet(M)
            J_frob = torch.linalg.norm(J, ord='fro', dim=(-2, -1))

            # Hutchinson estimates at all probe counts
            hutch = {}
            for n_p in probe_counts:
                hutch[n_p] = hutchinson_trace_batched(
                    vel_fn, a_flat, t, cond["state"], n_p,
                    B, D, horizon_steps, action_dim, device,
                )

            all_results_per_step[k].append({
                "exact_trace": exact_trace,
                "exact_logdet": exact_logdet,
                "J_frob": J_frob,
                "hutch": hutch,
            })

            # advance ODE using combined velocity (always)
            v = model._combined_velocity(a, t, cond)
            a = a + v * dt

    # print results
    for k in range(K - 1):
        results = all_results_per_step[k]
        exact_traces = torch.cat([r["exact_trace"] for r in results])
        exact_logdets = torch.cat([r["exact_logdet"] for r in results])
        J_frobs = torch.cat([r["J_frob"] for r in results])
        taylor_1st = exact_traces * dt

        N = exact_traces.shape[0]
        print(f"\n  Step k={k}, t={k*dt:.2f}  (N={N} samples)")
        print(f"    ||J||_F:  mean={J_frobs.mean():.3f}  std={J_frobs.std():.3f}  "
              f"||J*dt||_F={J_frobs.mean()*dt:.3f}")
        print(f"    tr(J):    mean={exact_traces.mean():.4f}  "
              f"std={exact_traces.std():.4f}")
        print(f"    logdet:   mean={exact_logdets.mean():.4f}  "
              f"std={exact_logdets.std():.4f}")

        taylor_err = taylor_1st - exact_logdets
        rel_denom = exact_logdets.abs().clamp(min=1e-6)
        print(f"    Taylor bias (tr*dt - logdet): "
              f"mean={taylor_err.mean():.6f}  std={taylor_err.std():.6f}")
        print(f"    Taylor |rel err|: "
              f"mean={(taylor_err/rel_denom).abs().mean():.4f}  "
              f"p95={(taylor_err/rel_denom).abs().quantile(0.95):.4f}")

        print(f"\n    {'n_probes':>8}  {'trace_bias':>11}  {'trace_rmse':>11}  "
              f"{'trace_rel%':>10}  {'logdet_rmse':>12}  {'logdet_rel%':>11}")
        print(f"    {'-'*8}  {'-'*11}  {'-'*11}  {'-'*10}  {'-'*12}  {'-'*11}")

        for n_p in probe_counts:
            ht = torch.cat([r["hutch"][n_p] for r in results])
            tr_err = ht - exact_traces
            ld_err = ht * dt - exact_logdets

            tr_rel_pct = 100 * tr_err.pow(2).mean().sqrt() / exact_traces.abs().mean().clamp(min=1e-8)
            ld_rel_pct = 100 * ld_err.pow(2).mean().sqrt() / exact_logdets.abs().mean().clamp(min=1e-8)

            print(f"    {n_p:>8}  {tr_err.mean():>11.4f}  "
                  f"{tr_err.pow(2).mean().sqrt():>11.4f}  "
                  f"{tr_rel_pct:>9.1f}%  "
                  f"{ld_err.pow(2).mean().sqrt():>12.6f}  "
                  f"{ld_rel_pct:>10.1f}%")


@torch.no_grad()
def test_accuracy(model, device, obs_pool, B=64, n_trials=100, label=""):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  D={model.act_dim_total}, K={model.inference_steps}, "
          f"dt={1.0/model.inference_steps}, B={B}, n_trials={n_trials}")
    print(f"{'='*70}")

    # J_combined
    test_jacobian(
        model, device,
        vel_fn=model._combined_velocity,
        jacrev_fn=model._per_sample_combined_velocity_flat,
        jac_name="J_combined = d(v_base + v_res)/da",
        obs_pool=obs_pool, B=B, n_trials=n_trials,
    )

    # J_base
    def base_vel(a, t, cond):
        return model.v_base(a, t, cond)

    test_jacobian(
        model, device,
        vel_fn=base_vel,
        jacrev_fn=model._per_sample_base_velocity_flat,
        jac_name="J_base = d(v_base)/da",
        obs_pool=obs_pool, B=B, n_trials=n_trials,
    )

    # J_res
    def res_vel(a, t, cond):
        return model.v_res(a, t, cond)

    test_jacobian(
        model, device,
        vel_fn=res_vel,
        jacrev_fn=model._per_sample_residual_velocity_flat,
        jac_name="J_res = d(v_res)/da",
        obs_pool=obs_pool, B=B, n_trials=n_trials,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="A")
    parser.add_argument("--iters", type=str, default="0,100000,199999")
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_map = {"A": "ablation_runA_baseline", "B": "ablation_runB_reward_penalty",
               "C": "ablation_runC_hutchinson", "D": "ablation_runD_detach_logdet",
               "E": "ablation_runE_vres_l2"}
    run_name = run_map[args.run]
    ckpt_dir = f"/home/tw559/steering_pi/KL_ReinFlow/log/gym/finetune/{run_name}/checkpoint"
    device = torch.device(args.device)

    model = make_model(device)
    model.eval()

    obs_pool = load_real_obs(device, n_obs=1024)
    print(f"Loaded {obs_pool.shape[0]} real observations from D4RL")
    print(f"  obs range: [{obs_pool.min():.3f}, {obs_pool.max():.3f}]")

    iters = [int(x) for x in args.iters.split(",")]
    torch.manual_seed(42)

    for it in iters:
        ckpt_path = f"{ckpt_dir}/state_{it}.pt"
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found, skipping")
            continue

        data = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(data["model"])
        model.eval()

        test_accuracy(model, device, obs_pool, B=args.B, n_trials=args.n_trials,
                      label=f"Checkpoint iter {it} ({run_name})")


if __name__ == "__main__":
    main()
