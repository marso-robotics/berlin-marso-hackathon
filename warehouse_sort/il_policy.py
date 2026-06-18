"""Imitation-learning policy entrypoints for ``eval.py`` / ``test.py``.

These adapt checkpoints produced by the vendored ManiSkill IL baselines
(``il/baselines/bc`` and ``il/baselines/act``) to the project's policy contract:

    policy.act(obs, deterministic=True) -> Tensor (num_envs, action_dim) in [-1, 1]

where ``obs`` is EXACTLY the env's observation in the difficulty's locked obs mode (a flat
state tensor for easy; a ``{"rgb","state"}`` dict for the rgb milestone). Wire one in via the
config ``policy`` field, e.g.:

    pixi run python eval.py difficulty=easy obs_mode=state \
        policy=warehouse_sort.il_policy:load_bc \
        checkpoint=il/baselines/bc/runs/<run>/checkpoints/best_eval_success_at_end.pt \
        eval_config=conf/eval/default.yaml

The architectures here MUST match the baselines' so the saved ``state_dict`` loads cleanly.
"""

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# State BC: the exact MLP from il/baselines/bc/bc.py (Actor).
# --------------------------------------------------------------------------- #
class BCActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state):
        return self.net(state)


class _StatePolicy:
    """Wraps a state->action net to satisfy the .act contract (deterministic BC: no sampling)."""

    def __init__(self, net, device):
        self.net = net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        if isinstance(obs, dict):           # tolerate {"state": ...} dicts
            obs = obs["state"]
        obs = obs.float().to(self.device)
        return self.net(obs).clamp(-1.0, 1.0)


def load_bc(checkpoint, sample_obs, action_space, device):
    """Entrypoint for the state-MLP BC checkpoint ({"actor": state_dict})."""
    state = sample_obs["state"] if isinstance(sample_obs, dict) else sample_obs
    state_dim = state.shape[1]
    action_dim = action_space.shape[0]
    net = BCActor(state_dim, action_dim)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["actor"])
    return _StatePolicy(net, device)


# --------------------------------------------------------------------------- #
# Diffusion Policy (state). Deployed FULLY CLOSED-LOOP: every step we re-run the
# diffusion sampler from a short obs history and execute only the first predicted action
# (i.e. action horizon = 1 <= prediction horizon). This is the most robust deployment, is
# stateless across episodes (no chunk queue to leak between rollouts), and matches the
# "execute fewer steps than predicted" option. Architecture/horizons must match training.
# --------------------------------------------------------------------------- #
def _add_baseline_path(rel):
    import os, sys
    p = os.path.join(os.path.dirname(__file__), "..", "il", "baselines", rel)
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)


class _DPPolicy:
    def __init__(self, net, scheduler, obs_horizon, pred_horizon, act_dim, device,
                 num_inference_steps=16):
        self.net = net.to(device).eval()
        self.scheduler = scheduler
        # DP works well with far fewer denoising steps at inference than the 100 train steps;
        # 16 keeps quality and makes per-step re-querying ~6x cheaper for the judged eval.
        self.scheduler.set_timesteps(num_inference_steps)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.device = device
        self.prev = None

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        cur = (obs["state"] if isinstance(obs, dict) else obs).float().to(self.device)
        if self.prev is None or self.prev.shape != cur.shape:
            self.prev = cur
        # obs_horizon-frame history (repeat earliest if needed); newest last
        hist = [self.prev, cur][-self.obs_horizon:]
        while len(hist) < self.obs_horizon:
            hist = [hist[0]] + hist
        self.prev = cur
        obs_cond = torch.stack(hist, dim=1).flatten(start_dim=1)  # (N, obs_horizon*obs_dim)
        B = cur.shape[0]
        naction = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)
        for k in self.scheduler.timesteps:
            noise_pred = self.net(sample=naction, timestep=k, global_cond=obs_cond)
            naction = self.scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        return naction[:, self.obs_horizon - 1].clamp(-1.0, 1.0)  # execute first action only


def load_dp(checkpoint, sample_obs, action_space, device,
            obs_horizon=2, pred_horizon=16, diffusion_step_embed_dim=64,
            unet_dims=(64, 128, 256), n_groups=8, num_diffusion_iters=100,
            num_inference_steps=16):
    """Entrypoint for a Diffusion Policy state checkpoint (uses the EMA weights)."""
    _add_baseline_path("diffusion_policy")
    from diffusion_policy.conditional_unet1d import ConditionalUnet1D
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    import numpy as np

    state = sample_obs["state"] if isinstance(sample_obs, dict) else sample_obs
    obs_dim = state.shape[1]
    act_dim = action_space.shape[0]
    net = ConditionalUnet1D(
        input_dim=act_dim, global_cond_dim=obs_horizon * obs_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=list(unet_dims), n_groups=n_groups,
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("ema_agent", ckpt.get("agent"))
    # ema_agent/agent state_dict keys are prefixed with "noise_pred_net."
    net_sd = {k.replace("noise_pred_net.", "", 1): v for k, v in sd.items()
              if k.startswith("noise_pred_net.")}
    net.load_state_dict(net_sd)
    scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_iters,
                              beta_schedule="squaredcos_cap_v2", clip_sample=True,
                              prediction_type="epsilon")
    return _DPPolicy(net, scheduler, obs_horizon, pred_horizon, act_dim, device,
                     num_inference_steps=num_inference_steps)


# --------------------------------------------------------------------------- #
# RGB Diffusion Policy (wrist cam + robot proprioception, NO privileged state, NO depth).
# Same fixed image input shape at every difficulty, so the SAME checkpoint runs across configs.
# Reuses the vendored train_rgbd Agent (PlainConv encoder + U-Net); deployed fully closed-loop.
# --------------------------------------------------------------------------- #
class _DPRgbPolicy:
    def __init__(self, agent, obs_horizon, device, num_inference_steps=16):
        self.agent = agent.to(device).eval()
        self.agent.noise_scheduler.set_timesteps(num_inference_steps)
        self.obs_horizon = obs_horizon
        self.device = device
        self.prev = None

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        state = obs["state"].float().to(self.device)
        rgb = obs["rgb"].to(self.device)
        cur = {"state": state, "rgb": rgb}
        if self.prev is None or self.prev["state"].shape != state.shape:
            self.prev = cur
        # obs_horizon history (newest last); obs_horizon=2 -> [prev, cur]
        obs_seq = {
            "state": torch.stack([self.prev["state"], state], dim=1),
            "rgb": torch.stack([self.prev["rgb"], rgb], dim=1),
        }
        self.prev = cur
        aseq = self.agent.get_action(obs_seq)          # (N, act_horizon, act_dim)
        return aseq[:, 0].clamp(-1.0, 1.0)             # execute first predicted action only


def load_dp_rgb(checkpoint, sample_obs, action_space, device,
                obs_horizon=2, act_horizon=8, pred_horizon=16,
                diffusion_step_embed_dim=64, unet_dims=(64, 128, 256), n_groups=8,
                num_inference_steps=16):
    """Entrypoint for an RGB Diffusion Policy checkpoint (vendored train_rgbd; uses EMA weights).

    ``sample_obs`` is the locked rgb obs dict {"rgb": (N,H,W,3) uint8, "state": (N,S) proprio}.
    """
    import types
    import numpy as np
    import gymnasium.spaces as spaces
    _add_baseline_path("diffusion_policy")
    from train_rgbd import Agent  # vendored RGB DP agent (PlainConv + ConditionalUnet1D)

    h, w, c = sample_obs["rgb"].shape[1:]
    state_dim = sample_obs["state"].shape[1]
    # stub env exposing the spaces the Agent reads (per-frame shapes are (obs_horizon, ...))
    stub = types.SimpleNamespace(
        single_observation_space=spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (obs_horizon, state_dim), np.float32),
            "rgb": spaces.Box(0, 255, (obs_horizon, h, w, c), np.uint8),
        }),
        single_action_space=spaces.Box(-1.0, 1.0, (action_space.shape[0],), np.float32),
    )
    args = types.SimpleNamespace(
        obs_horizon=obs_horizon, act_horizon=act_horizon, pred_horizon=pred_horizon,
        diffusion_step_embed_dim=diffusion_step_embed_dim, unet_dims=list(unet_dims),
        n_groups=n_groups,
    )
    agent = Agent(stub, args)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt.get("ema_agent", ckpt.get("agent")))
    return _DPRgbPolicy(agent, obs_horizon, device, num_inference_steps=num_inference_steps)
