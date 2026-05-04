# MIT License
# Copyright (c) 2025 ReinFlow Authors
"""
SAC fine-tuning of a flow-matching policy with a residual velocity head.

Architecture (per New Project Idea 0501):
  a^0 ~ N(0, I)
  for k = 0..K-2:                                         # deterministic ODE
    a^{k+1} = a^k + (v_base + v_res)(a^k, t_k, s) * dt
  eps ~ N(0, I)
  a^K = a^{K-1} + (v_base + v_res)(a^{K-1}, t_{K-1}, s) * dt + sigma_phi(s) * eps

Losses
  L_critic = MSE(Q(s, a^K), r + gamma * (1-d) * min Q_target(s', a'^K))   (no entropy term)
  L_actor  = -min(Q1, Q2)(s, a^K)                                        (reparameterized; no log pi)
  L_KL     = E [ log p_theta^det(a^{K-1}) - log p_base^det(a^{K-1}) ]    (exact, change of variables)
  L_jac    = (1/(K-1)) * sum_k || dv_res/da ||_F^2 * dt                  (Frobenius reg on v_res only)
"""
import logging
import copy
from typing import Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.func import jacrev, vmap
from torch.distributions import Normal

from model.flow.mlp_flow import FlowMLP

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# observation-conditioned exploration sigma head
# -----------------------------------------------------------------------------
class SigmaHead(nn.Module):
    """sigma_phi(o), bounded to [sigma_min, sigma_max] via sigmoid."""

    def __init__(self, cond_dim, action_dim, horizon_steps,
                 hidden_dims=(64, 64), sigma_min=0.05, sigma_max=0.15):
        super().__init__()
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        out_dim = action_dim * horizon_steps
        layers = []
        in_dim = cond_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.SiLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        # initialize last layer to produce ~mid range
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: Tensor) -> Tensor:
        # state: (B, To, Do) or (B, To*Do)
        B = state.shape[0]
        x = state.view(B, -1)
        s = torch.sigmoid(self.net(x))
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * s
        return sigma.view(B, self.horizon_steps, self.action_dim)


# -----------------------------------------------------------------------------
# Main module
# -----------------------------------------------------------------------------
class SACResidualFlow(nn.Module):
    def __init__(
        self,
        device,
        base_policy: FlowMLP,        # frozen v_base
        residual_policy: FlowMLP,    # trainable v_res (zero-init last layer)
        critic: nn.Module,           # CriticObsAct (double Q)
        sigma_head: SigmaHead,
        actor_policy_path: str,
        act_dim: int,
        horizon_steps: int,
        act_min: float,
        act_max: float,
        obs_dim: int,
        cond_steps: int,
        inference_steps: int,
        denoised_clip_value: float = 1.0,
        randn_clip_value: float = 3.0,
        backward_fp_iters: int = 10,
        alpha: float = 0.1,
        kl_weight: float = 0.05,
        jac_weight: float = 0.01,
        sigma_entropy_weight: float = 0.01,
        target_ema_rate: float = 0.005,
        zero_init_residual: bool = True,
        kl_mode: str = "exact",
        vres_l2_weight: float = 0.0,
        hutchinson_samples: int = 1,
    ):
        super().__init__()
        self.device = device
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = act_dim * horizon_steps
        self.act_min = act_min
        self.act_max = act_max
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        self.inference_steps = inference_steps
        self.denoised_clip_value = denoised_clip_value
        self.randn_clip_value = randn_clip_value
        self.backward_fp_iters = backward_fp_iters
        self.alpha = alpha
        self.kl_weight = kl_weight
        self.jac_weight = jac_weight
        self.sigma_entropy_weight = sigma_entropy_weight
        self.target_ema_rate = target_ema_rate
        self.kl_mode = kl_mode
        self.vres_l2_weight = vres_l2_weight
        self.hutchinson_samples = hutchinson_samples

        # frozen base
        self.v_base: FlowMLP = base_policy.to(device)
        self._load_base_policy(actor_policy_path, use_ema=True)
        for p in self.v_base.parameters():
            p.requires_grad = False
        self.v_base.eval()

        # trainable residual (initialize to zero so we start from base policy behavior)
        self.v_res: FlowMLP = residual_policy.to(device)
        if zero_init_residual:
            self._zero_init_velocity_head(self.v_res)

        # twin critic + targets
        self.critic = critic.to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # sigma head (obs-conditioned, bounded)
        self.sigma_head = sigma_head.to(device)

        self._report_params()

    # ------------------------------------------------------------------ utils
    def _load_base_policy(self, path, use_ema=True):
        if not path:
            log.warning("No base policy path; using randomly init base. KL/Jac will be uninformative.")
            return
        log.info(f"Loading base policy from {path}")
        data = torch.load(path, map_location=self.device, weights_only=True)
        key = "ema" if (use_ema and "ema" in data) else "model"
        sd = {k.replace("network.", ""): v for k, v in data[key].items()}
        self.v_base.load_state_dict(sd)
        log.info(f"Loaded base policy ({key}).")

    @staticmethod
    def _zero_init_velocity_head(flow: FlowMLP):
        """Zero the last linear layer of the velocity head so v_res(a, t, s) ~= 0 at init."""
        for m in reversed(list(flow.mlp_mean.modules())):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                break

    def _report_params(self):
        n_base = sum(p.numel() for p in self.v_base.parameters()) / 1e6
        n_res = sum(p.numel() for p in self.v_res.parameters()) / 1e6
        n_crit = sum(p.numel() for p in self.critic.parameters()) / 1e6
        n_sig = sum(p.numel() for p in self.sigma_head.parameters()) / 1e6
        log.info(f"Params (M): v_base(frozen)={n_base:.3f}  v_res={n_res:.3f}  "
                 f"critic={n_crit:.3f}  sigma={n_sig:.3f}")

    def _combined_velocity(self, a: Tensor, t: Tensor, cond: Dict[str, Tensor]) -> Tensor:
        return self.v_base(a, t, cond) + self.v_res(a, t, cond)

    def update_target_critic(self, tau=None):
        tau = self.target_ema_rate if tau is None else tau
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    # --------------------------------------------------------------- sampling
    def sample_action(
        self,
        cond: Dict[str, Tensor],
        deterministic: bool = False,
        return_intermediate: bool = False,
        return_trajectory: bool = False,
    ):
        """Forward ODE (deterministic, residual active) + final noisy step.

        Reparameterized: gradients flow through both the chain and through eps via sigma_phi.

        Returns:
            a_K  : (B, Ta, Da)        executed action
            a_Km1: (B, Ta, Da)        penultimate point                       (if return_intermediate)
            eps  : (B, Ta, Da)        Gaussian noise sample at last step      (if return_intermediate)
            traj_a: list of (B, Ta, Da)  ODE trajectory points                (if return_trajectory)
            traj_t: list of (B,)                                              (if return_trajectory)
        """
        B = cond["state"].shape[0]
        device = self.device
        K = self.inference_steps
        dt = 1.0 / K

        a = torch.randn(B, self.horizon_steps, self.action_dim, device=device)

        traj_a = []
        traj_t = []

        for k in range(K - 1):
            t = torch.full((B,), k * dt, device=device)
            if return_trajectory:
                traj_a.append(a)
                traj_t.append(t)
            v = self._combined_velocity(a, t, cond)
            a = a + v * dt

        a_Km1 = a

        t_last = torch.full((B,), (K - 1) * dt, device=device)
        v_last = self._combined_velocity(a_Km1, t_last, cond)
        sigma = self.sigma_head(cond["state"])  # (B, Ta, Da)
        if deterministic:
            eps = torch.zeros_like(a_Km1)
        else:
            eps = torch.randn_like(a_Km1)
            eps = eps.clamp(-self.randn_clip_value, self.randn_clip_value)
        a_K = a_Km1 + v_last * dt + sigma * eps
        a_K = a_K.clamp(self.act_min, self.act_max)

        if return_intermediate and return_trajectory:
            return a_K, a_Km1, eps, traj_a, traj_t
        if return_intermediate:
            return a_K, a_Km1, eps
        return a_K

    # ---------------------- forward log p_theta with full Jacobians (per step)
    def _per_sample_combined_velocity_flat(self, a_flat, t_scalar, cond_state):
        """Single-sample combined velocity, output flattened to (D,).

        Used inside vmap(jacrev(...)) to obtain per-sample Jacobian (D, D).
        """
        a = a_flat.view(self.horizon_steps, self.action_dim).unsqueeze(0)  # (1, Ta, Da)
        t = t_scalar.unsqueeze(0)
        cond = {"state": cond_state.unsqueeze(0)}
        v = self._combined_velocity(a, t, cond).squeeze(0).flatten()
        return v

    def _per_sample_base_velocity_flat(self, a_flat, t_scalar, cond_state):
        a = a_flat.view(self.horizon_steps, self.action_dim).unsqueeze(0)
        t = t_scalar.unsqueeze(0)
        cond = {"state": cond_state.unsqueeze(0)}
        v = self.v_base(a, t, cond).squeeze(0).flatten()
        return v

    def _per_sample_residual_velocity_flat(self, a_flat, t_scalar, cond_state):
        a = a_flat.view(self.horizon_steps, self.action_dim).unsqueeze(0)
        t = t_scalar.unsqueeze(0)
        cond = {"state": cond_state.unsqueeze(0)}
        v = self.v_res(a, t, cond).squeeze(0).flatten()
        return v

    def forward_with_logdet(self, cond: Dict[str, Tensor]):
        """Run K-1 deterministic steps with v_base + v_res while accumulating log|det(I + J*dt)|.

        Returns:
            a_Km1   : (B, Ta, Da)
            log_p   : (B,)                 log p_theta^det at a^{K-1}
            traj_a  : list of (B, Ta, Da)  trajectory for Jacobian-reg reuse
            traj_t  : list of (B,)
        """
        B = cond["state"].shape[0]
        device = self.device
        K = self.inference_steps
        dt = 1.0 / K
        D = self.act_dim_total
        I_D = torch.eye(D, device=device).unsqueeze(0)

        a = torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        log_p0 = Normal(torch.zeros_like(a), 1.0).log_prob(a).sum(dim=(-2, -1))
        sum_logdet = torch.zeros(B, device=device)

        traj_a = []
        traj_t = []

        for k in range(K - 1):
            t = torch.full((B,), k * dt, device=device)
            traj_a.append(a)
            traj_t.append(t)

            # Per-sample Jacobian via vmap(jacrev): J in (B, D, D)
            J = vmap(jacrev(self._per_sample_combined_velocity_flat, argnums=0))(
                a.view(B, D), t, cond["state"]
            )
            M = I_D + J * dt                   # (B, D, D)
            sign, logabsdet = torch.linalg.slogdet(M)
            # If sign goes non-positive somewhere, the discrete map stops being a diffeomorphism.
            # We still proceed but log it.
            sum_logdet = sum_logdet + logabsdet

            # Advance state. NO intermediate clamping (would break the change of variables;
            # see sample_action note).
            v = self._combined_velocity(a, t, cond)
            a = a + v * dt

        log_p = log_p0 - sum_logdet
        return a, log_p, traj_a, traj_t

    def forward_with_hutchinson_logdet(self, cond: Dict[str, Tensor]):
        """Forward ODE with Hutchinson trace estimator for log|det|.

        Uses log|det(I + J*dt)| ≈ tr(J)*dt, estimated via v^T J v with Rademacher v.
        Only requires JVPs — no full Jacobian or slogdet, so backward is clean.
        """
        B = cond["state"].shape[0]
        device = self.device
        K = self.inference_steps
        dt = 1.0 / K
        D = self.act_dim_total

        a = torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        log_p0 = Normal(torch.zeros_like(a), 1.0).log_prob(a).sum(dim=(-2, -1))
        sum_logdet = torch.zeros(B, device=device)

        traj_a = []
        traj_t = []

        for k in range(K - 1):
            t = torch.full((B,), k * dt, device=device)
            traj_a.append(a)
            traj_t.append(t)

            a_flat = a.view(B, D)
            tr_est = torch.zeros(B, device=device)
            for _ in range(self.hutchinson_samples):
                v_probe = torch.randint(0, 2, (B, D), device=device).float() * 2 - 1
                def vel_fn(a_f):
                    aa = a_f.view(B, self.horizon_steps, self.action_dim)
                    tt = t
                    return self._combined_velocity(aa, tt, cond).view(B, D)
                Jv = torch.func.jvp(vel_fn, (a_flat,), (v_probe,))[1]
                tr_est = tr_est + (v_probe * Jv).sum(dim=-1)
            tr_est = tr_est / self.hutchinson_samples
            sum_logdet = sum_logdet + tr_est * dt

            v = self._combined_velocity(a, t, cond)
            a = a + v * dt

        log_p = log_p0 - sum_logdet
        return a, log_p, traj_a, traj_t

    def forward_with_detached_logdet(self, cond: Dict[str, Tensor]):
        """Forward ODE with exact logdet value but detached from gradient graph.

        Computes exact log|det(I + J*dt)| for monitoring, but detaches it so
        gradient only flows through log_p0 and trajectory (no slogdet backward).
        """
        B = cond["state"].shape[0]
        device = self.device
        K = self.inference_steps
        dt = 1.0 / K
        D = self.act_dim_total
        I_D = torch.eye(D, device=device).unsqueeze(0)

        a = torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        log_p0 = Normal(torch.zeros_like(a), 1.0).log_prob(a).sum(dim=(-2, -1))
        sum_logdet = torch.zeros(B, device=device)

        traj_a = []
        traj_t = []

        for k in range(K - 1):
            t = torch.full((B,), k * dt, device=device)
            traj_a.append(a)
            traj_t.append(t)

            with torch.no_grad():
                J = vmap(jacrev(self._per_sample_combined_velocity_flat, argnums=0))(
                    a.view(B, D), t, cond["state"]
                )
                M = I_D + J * dt
                _, logabsdet = torch.linalg.slogdet(M)
            sum_logdet = sum_logdet + logabsdet

            v = self._combined_velocity(a, t, cond)
            a = a + v * dt

        log_p = log_p0 - sum_logdet
        return a, log_p, traj_a, traj_t

    # -------------------- backward base ODE with fixed-point iteration --------
    @torch.no_grad()
    def _backward_base_recover_trajectory(self, a_Km1: Tensor, cond: Dict[str, Tensor]):
        """Given a^{K-1}_theta, find what trajectory the base ODE would have taken to reach it.

        Implicit step: a^k = a^{k+1} - dt * v_base(a^k, t_k)
        Solved by fixed-point iteration (per pi0.5 verification doc).
        Evaluates Jacobians at the inverted point a^k.

        Returns:
            traj_base_a : list of (B, Ta, Da), length K-1, ordered [a^0_base, a^1_base, ..., a^{K-2}_base]
            traj_base_t : list of (B,)
        """
        B = a_Km1.shape[0]
        device = self.device
        K = self.inference_steps
        dt = 1.0 / K

        a_curr = a_Km1                         # this corresponds to index K-1 at start
        rev_traj_a = []                         # will collect a^{K-2}, a^{K-3}, ..., a^0
        rev_traj_t = []                         # corresponding t indices for J evaluation

        for k in range(K - 2, -1, -1):
            t_k = torch.full((B,), k * dt, device=device)
            # explicit estimate
            v_explicit = self.v_base(a_curr, t_k, cond)
            a_prev = a_curr - dt * v_explicit
            # fixed-point refine
            for _ in range(self.backward_fp_iters):
                v_fp = self.v_base(a_prev, t_k, cond)
                a_prev = a_curr - dt * v_fp
            rev_traj_a.append(a_prev)
            rev_traj_t.append(t_k)
            a_curr = a_prev

        # reverse to [a^0_base, a^1_base, ..., a^{K-2}_base]
        traj_a = list(reversed(rev_traj_a))
        traj_t = list(reversed(rev_traj_t))
        return traj_a, traj_t

    def backward_base_logprob(self, a_Km1: Tensor, cond: Dict[str, Tensor]) -> Tensor:
        """Compute log p_base^det(a^{K-1}) where a^{K-1} = a_Km1.

        log p_base = log p_0(a^0_base) - sum_{k=0..K-2} log|det(I + J_base(a^k_base, t_k) * dt)|

        Backward solve uses fixed-point iteration; Jacobians are evaluated along the recovered
        base trajectory (not at a^{k+1}_base). Frozen v_base => no gradient through these ops
        except via the input a_Km1.
        """
        B = a_Km1.shape[0]
        device = self.device
        K = self.inference_steps
        dt = 1.0 / K
        D = self.act_dim_total
        I_D = torch.eye(D, device=device).unsqueeze(0)

        # Fixed-point inversion is no-grad; the gradient w.r.t. a_Km1 enters via a^0_base only
        # through log p_0(a^0_base) which IS differentiable. To preserve that, we re-run the
        # final FP step under grad once.
        traj_a_nograd, traj_t = self._backward_base_recover_trajectory(a_Km1, cond)

        # Re-derive a^0_base differentiably from the no-grad trajectory:
        # repeat the FP equation once with grad enabled, starting from the converged neighbor.
        # This propagates only the linear sensitivity to a_Km1 through the chain.
        traj_a = []
        a_next = a_Km1
        for k in range(K - 2, -1, -1):
            a_neigh = traj_a_nograd[k]
            t_k = torch.full((B,), k * dt, device=device)
            v_at_neigh = self.v_base(a_neigh, t_k, cond)
            a_prev = a_next - dt * v_at_neigh    # one differentiable refinement
            traj_a.insert(0, a_prev)
            a_next = a_prev

        # log p_0 at a^0_base  (manual Gaussian to avoid Normal's finite-value check;
        # FP inversion can diverge on out-of-distribution samples)
        a0 = traj_a[0]
        log_p0_elementwise = -0.5 * (a0 ** 2 + 1.8378770664093453)  # log(2*pi)
        log_p0 = torch.nan_to_num(log_p0_elementwise, nan=0.0, posinf=0.0, neginf=-1e4).sum(dim=(-2, -1))

        # Forward base log-dets evaluated at the recovered base trajectory
        sum_logdet = torch.zeros(B, device=device)
        for k in range(K - 1):
            a_k = traj_a[k]
            t_k = traj_t[k]
            J = vmap(jacrev(self._per_sample_base_velocity_flat, argnums=0))(
                a_k.view(B, D), t_k, cond["state"]
            )
            M = I_D + J * dt
            _, logabsdet = torch.linalg.slogdet(M)
            sum_logdet = sum_logdet + logabsdet

        return log_p0 - sum_logdet

    # ------------------------------------------------------------- KL & Jac
    def compute_kl_and_action(self, cond: Dict[str, Tensor]):
        """Compute KL (using self.kl_mode) plus produce a fresh action and optional Jac reg.

        kl_mode options:
          "exact"         — original: full Jacobian + slogdet (dangerous backward)
          "hutchinson"    — Hutchinson trace estimator (JVP only, clean backward)
          "detach_logdet" — exact logdet value, but detached (no slogdet gradient)

        Returns:
            a_K        : (B, Ta, Da)
            kl         : (B,)
            jac_reg    : scalar
            log_p_theta: (B,)
            log_p_base : (B,)
            eps        : (B, Ta, Da)  noise sample at last step
            sigma      : (B, Ta, Da)  sigma_head output
        """
        B = cond["state"].shape[0]
        device = self.device
        K = self.inference_steps
        dt = 1.0 / K
        D = self.act_dim_total

        if self.kl_mode == "hutchinson":
            a_Km1, log_p_theta, traj_a, traj_t = self.forward_with_hutchinson_logdet(cond)
        elif self.kl_mode == "detach_logdet":
            a_Km1, log_p_theta, traj_a, traj_t = self.forward_with_detached_logdet(cond)
        else:
            a_Km1, log_p_theta, traj_a, traj_t = self.forward_with_logdet(cond)

        # For hutchinson/detach_logdet: detach log_p_base to avoid Path 3b
        # (slogdet backward in backward_base_logprob). KL gradient comes
        # entirely from log_p_theta side. log_p_base is still computed for
        # the KL value (monitoring) but contributes no gradient.
        detach_base = self.kl_mode in ("hutchinson", "detach_logdet")
        if detach_base:
            with torch.no_grad():
                log_p_base = self.backward_base_logprob(a_Km1, cond)
            log_p_base = log_p_base.detach()
        else:
            log_p_base = self.backward_base_logprob(a_Km1, cond)

        kl_raw = log_p_theta - log_p_base
        kl = torch.nan_to_num(kl_raw, nan=0.0, posinf=50.0, neginf=-50.0)
        kl = kl.clamp(-50.0, 50.0)

        jac_reg = a_Km1.new_zeros(())
        if self.jac_weight > 0:
            for k in range(K - 1):
                a_k = traj_a[k]
                t_k = traj_t[k]
                J_res = vmap(jacrev(self._per_sample_residual_velocity_flat, argnums=0))(
                    a_k.view(B, D), t_k, cond["state"]
                )
                jac_reg = jac_reg + (J_res ** 2).sum(dim=(-2, -1)).mean() * dt
            jac_reg = jac_reg / (K - 1)

        t_last = torch.full((B,), (K - 1) * dt, device=device)
        v_last = self._combined_velocity(a_Km1, t_last, cond)
        sigma = self.sigma_head(cond["state"])
        eps = torch.randn_like(a_Km1).clamp(-self.randn_clip_value, self.randn_clip_value)
        a_K = (a_Km1 + v_last * dt + sigma * eps).clamp(self.act_min, self.act_max)

        return a_K, kl, jac_reg, log_p_theta, log_p_base, eps, sigma

    @torch.no_grad()
    def compute_kl_for_reward(self, cond: Dict[str, Tensor]) -> Tensor:
        """Compute per-sample KL estimate for reward penalty. No gradients.

        Draws fresh noise and runs forward+backward to estimate E_z[KL(z)].
        Cannot use the executed trajectory's a_Km1 because computing log_p_theta
        at an arbitrary point requires backward-inverting the combined ODE (same
        stiff fixed-point problem we're trying to avoid).
        """
        a_Km1, log_p_theta, _, _ = self.forward_with_logdet(cond)
        log_p_base = self.backward_base_logprob(a_Km1, cond)
        kl_raw = log_p_theta - log_p_base
        kl = torch.nan_to_num(kl_raw, nan=0.0, posinf=50.0, neginf=-50.0)
        return kl.clamp(-50.0, 50.0)

    def compute_vres_l2(self, cond: Dict[str, Tensor], traj_a, traj_t):
        """||v_res||^2 averaged over trajectory steps. Simple proxy for KL.

        Detaches trajectory points so gradient only flows through v_res params,
        not back through the ODE that produced the trajectory.
        """
        penalty = traj_a[0].new_zeros(())
        for k in range(len(traj_a)):
            v_k = self.v_res(traj_a[k].detach(), traj_t[k], cond)
            penalty = penalty + (v_k ** 2).sum(dim=(-2, -1)).mean()
        return penalty / max(len(traj_a), 1)

    # ----------------------------------------------------------------- losses
    def loss_critic(self, obs, next_obs, actions, rewards, terminated, gamma):
        with torch.no_grad():
            a_next = self.sample_action(next_obs, deterministic=False)
            q1_t, q2_t = self.target_critic(next_obs, a_next)
            q_t = torch.min(q1_t, q2_t)
            target = rewards + gamma * q_t * (1.0 - terminated)
        q1, q2 = self.critic(obs, actions)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        info = {
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "q_target_mean": target.mean().item(),
            "q_target_std": target.std().item(),
            "q_target_max": target.max().item(),
            "q_target_min": target.min().item(),
            "reward_batch_mean": rewards.mean().item(),
            "reward_batch_std": rewards.std().item(),
        }
        return loss, info

    def loss_actor(self, obs):
        """SAC actor loss with KL regularizer (mode-dependent) and entropy.

        All modes get SAC entropy: alpha * (-0.5*eps^2 - log(sigma) - 0.5*log(2*pi)).
        KL term depends on self.kl_mode:
          "none"           — no KL, just SAC + entropy (baseline)
          "reward_penalty" — KL handled in reward, not in actor loss
          "exact"          — full Jacobian + slogdet (original, dangerous backward)
          "hutchinson"     — trace estimator (clean backward)
          "detach_logdet"  — exact value, detached gradient (weak KL signal)
          "vres_l2"        — ||v_res||^2 proxy (no Jacobians)
        """
        import math

        use_kl_in_loss = (
            self.kl_mode not in ("none", "reward_penalty", "vres_l2")
            and (self.kl_weight > 0 or self.jac_weight > 0)
        )
        use_vres_l2 = self.kl_mode == "vres_l2" and self.vres_l2_weight > 0

        if use_kl_in_loss:
            a_K, kl, jac_reg, log_p_theta, log_p_base, eps, sigma = self.compute_kl_and_action(obs)
            kl_loss = self.kl_weight * kl.mean()
            jac_reg = torch.nan_to_num(jac_reg, nan=0.0, posinf=1e4, neginf=0.0)
            jac_loss = self.jac_weight * jac_reg
        else:
            use_traj = use_vres_l2
            if use_traj:
                a_K, a_Km1, eps, traj_a, traj_t = self.sample_action(
                    obs, return_intermediate=True, return_trajectory=True,
                )
            else:
                a_K, a_Km1, eps = self.sample_action(obs, return_intermediate=True)
            sigma = self.sigma_head(obs["state"])
            kl_loss = a_K.new_zeros(())
            jac_loss = a_K.new_zeros(())

        q1, q2 = self.critic(obs, a_K)
        q_min = torch.min(q1, q2)
        sac_loss = -q_min.mean()

        sigma_entropy = sigma.log().sum(dim=(-2, -1)).mean()
        sigma_ent_loss = -self.sigma_entropy_weight * sigma_entropy

        # SAC entropy: alpha * log_prob (all modes)
        # eps^2 is detached noise, so only -log(sigma) contributes gradient
        entropy_loss = a_K.new_zeros(())
        if self.alpha > 0:
            log_prob = (
                -0.5 * eps.pow(2) - sigma.log() - 0.5 * math.log(2.0 * math.pi)
            ).sum(dim=(-2, -1))
            entropy_loss = self.alpha * log_prob.mean()

        # v_res L2 penalty using trajectory from sample_action (no redundant ODE)
        vres_loss = a_K.new_zeros(())
        if use_vres_l2:
            vres_loss = self.vres_l2_weight * self.compute_vres_l2(obs, traj_a, traj_t)

        total = sac_loss + kl_loss + jac_loss + sigma_ent_loss + entropy_loss + vres_loss

        info = {
            "loss_sac": sac_loss.item(),
            "loss_kl": kl_loss.item(),
            "loss_jac": jac_loss.item(),
            "loss_sigma_ent": sigma_ent_loss.item(),
            "loss_entropy": entropy_loss.item(),
            "loss_vres_l2": vres_loss.item(),
            "q_mean": q_min.mean().item(),
            "sigma_mean": sigma.mean().item(),
            "sigma_min": sigma.min().item(),
            "sigma_max": sigma.max().item(),
            "action_mean": a_K.mean().item(),
            "action_std": a_K.std().item(),
            "action_abs_max": a_K.abs().max().item(),
        }
        if use_kl_in_loss:
            info.update({
                "kl_mean": kl.mean().item(),
                "kl_std": kl.std().item(),
                "kl_max": kl.max().item(),
                "log_p_theta_mean": log_p_theta.mean().item(),
                "log_p_theta_std": log_p_theta.std().item(),
                "log_p_base_mean": log_p_base.mean().item(),
                "log_p_base_std": log_p_base.std().item(),
            })
        return total, info
