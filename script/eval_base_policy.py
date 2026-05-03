"""Evaluate the base policy (v_base only, v_res=0) on Hopper-v2 for a few episodes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gym
import d4rl.gym_mujoco
from env.gym_utils.wrapper.mujoco_locomotion_lowdim import MujocoLocomotionLowdimWrapper
from model.flow.mlp_flow import FlowMLP

CKPT = "hf_cache/log/log_gym_d4rl_pretrained/hopper-medium-v2/1-ReFlow/state_40.pt"
NORM = "hf_cache/data-offline/gym_d4rl/hopper-medium-v2/normalization.npz"
K = 4
ACT_STEPS = 4
HORIZON_STEPS = 4
ACTION_DIM = 3
OBS_DIM = 11
N_EPISODES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base policy
v_base = FlowMLP(
    horizon_steps=HORIZON_STEPS, action_dim=ACTION_DIM,
    cond_dim=OBS_DIM, time_dim=16,
    mlp_dims=[512, 512, 512], activation_type="ReLU",
    out_activation_type="Identity", use_layernorm=False, residual_style=True,
)
data = torch.load(CKPT, map_location=device)
key = "ema" if "ema" in data else "model"
sd = {k.replace("network.", ""): v for k, v in data[key].items()}
v_base.load_state_dict(sd)
v_base = v_base.to(device)
v_base.eval()

# Create env
raw_env = gym.make("hopper-medium-v2")
env = MujocoLocomotionLowdimWrapper(raw_env, normalization_path=NORM)

dt = 1.0 / K

for ep in range(N_EPISODES):
    obs_dict = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < 1000:
        state = torch.from_numpy(obs_dict["state"]).float().to(device).unsqueeze(0).unsqueeze(0)
        cond = {"state": state}

        # Forward ODE: K steps, no residual, no noise
        with torch.no_grad():
            a = torch.randn(1, HORIZON_STEPS, ACTION_DIM, device=device)
            for k in range(K):
                t = torch.full((1,), k * dt, device=device)
                v = v_base(a, t, cond)
                a = a + v * dt
            a = a.clamp(-1.0, 1.0)

        action = a.cpu().numpy()[0]  # (HORIZON_STEPS, ACTION_DIM)

        # Execute each sub-action
        for step_i in range(ACT_STEPS):
            obs_dict, reward, done, info = env.step(action[step_i])
            total_reward += reward
            steps += 1
            if done:
                break

    print(f"Episode {ep}: reward={total_reward:.1f}, steps={steps}")

print(f"\nMean reward: {np.mean([0])}")  # placeholder, real mean below

# Re-run properly to collect
rewards = []
for ep in range(N_EPISODES):
    obs_dict = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < 1000:
        state = torch.from_numpy(obs_dict["state"]).float().to(device).unsqueeze(0).unsqueeze(0)
        cond = {"state": state}
        with torch.no_grad():
            a = torch.randn(1, HORIZON_STEPS, ACTION_DIM, device=device)
            for k in range(K):
                t = torch.full((1,), k * dt, device=device)
                v = v_base(a, t, cond)
                a = a + v * dt
            a = a.clamp(-1.0, 1.0)
        action = a.cpu().numpy()[0]
        for step_i in range(ACT_STEPS):
            obs_dict, reward, done, info = env.step(action[step_i])
            total_reward += reward
            steps += 1
            if done:
                break
    rewards.append(total_reward)

print(f"\nOver {N_EPISODES} episodes: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}, min={np.min(rewards):.1f}, max={np.max(rewards):.1f}")
