"""Stress test: verify backward_base_logprob doesn't crash on extreme observations."""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.flow.ft_sac.sac_residual_flow import SACResidualFlow, SigmaHead
from model.common.critic import CriticObsAct

device = torch.device("cpu")
B, K, D = 64, 4, 12
obs_dim, act_dim, act_steps = 11, 3, 4
ckpt = torch.load(
    "hf_cache/log/log_gym_d4rl_pretrained/hopper-medium-v2/1-ReFlow/state_40.pt",
    map_location="cpu",
)
model = SACResidualFlow(
    obs_dim=obs_dim, act_dim=act_dim, act_steps=act_steps,
    base_ckpt=ckpt, device=device,
    hidden_dims_base=[512, 512, 512], hidden_dims_res=[128, 128],
    inference_steps=K, act_min=-1.0, act_max=1.0,
    kl_weight=0.05, jac_weight=0.01,
)
sigma_head = SigmaHead(obs_dim=obs_dim, act_dim=act_dim, act_steps=act_steps, device=device)
critic = CriticObsAct(obs_dim=obs_dim, act_dim=act_dim * act_steps, hidden_dims=[256, 256, 256], device=device)
model.sigma_head = sigma_head
model.critic = critic

for trial, scale in enumerate([1.0, 10.0, 100.0, 1000.0]):
    obs = torch.randn(B, 1, obs_dim) * scale
    cond = {"state": obs}
    try:
        a_K, kl, jac_reg = model.compute_kl_and_action(cond)
        kl_finite = kl.isfinite().all().item()
        print(f"Trial {trial} (scale={scale:7.1f}): OK  "
              f"kl_mean={kl.mean().item():8.2f}  kl_max={kl.max().item():8.2f}  "
              f"kl_finite={kl_finite}  jac_reg={jac_reg.item():.4f}")
    except Exception as e:
        print(f"Trial {trial} (scale={scale:7.1f}): CRASHED: {type(e).__name__}: {e}")

print("\nStress test complete.")
