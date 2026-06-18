"""Shared helpers for the train/test/eval scripts: env construction, config + git-hash
logging. Keep this thin -- env/reward/policy logic lives in their own files."""

import subprocess
from typing import Optional

import gymnasium as gym
import torch
from omegaconf import OmegaConf

import warehouse_sort  # noqa: F401  (registers WarehouseSort-v1)
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# selects the env reward source from the `reward` config field (BUILD_SPEC §6)
REWARD_MODE = {"sparse": "sparse", "example_dense": "normalized_dense"}


def _gym_make(cfg, obs_mode, randomization, n, render_mode):
    """Construct the raw (unwrapped) ManiSkill env for the configured `env_id`.

    Default is the full WarehouseSort-v1 task (all difficulty/randomisation kwargs). The
    short-horizon SimpleSort-v1 ablation (one parcel, two bins, env+reward in one file) takes
    only the standard ManiSkill kwargs — none of WarehouseSort's parcel/randomisation knobs —
    so the same train/test/eval harness drives either env via `difficulty=simple`.
    """
    env_id = cfg.get("env_id", "WarehouseSort-v1")
    if env_id == "SimpleSort-v1":
        return gym.make(
            "SimpleSort-v1",
            num_envs=n,
            obs_mode=obs_mode,
            control_mode=cfg.control_mode,
            sim_backend="gpu",
            render_mode=render_mode,
            reward_mode=REWARD_MODE[cfg.reward],
            max_episode_steps=cfg.max_episode_steps,
        )
    return gym.make(
        "WarehouseSort-v1",
        num_envs=n,
        obs_mode=obs_mode,
        control_mode=cfg.control_mode,
        sim_backend="gpu",
        render_mode=render_mode,
        reward_mode=REWARD_MODE[cfg.reward],
        max_episode_steps=cfg.max_episode_steps,
        difficulty=cfg.difficulty.name,
        num_parcels=cfg.difficulty.num_parcels,
        fixed_poses=cfg.difficulty.fixed_poses,
        reward_option=cfg.reward,
        reward_impl=cfg.get("reward_impl", "example_dense"),
        camera_width=cfg.camera.width,
        camera_height=cfg.camera.height,
        obs_camera=cfg.get("obs_camera", "scene"),
        randomization=OmegaConf.to_container(randomization, resolve=True),
    )


def compose_cfg(overrides=None, config_dir=None):
    """Load the Hydra config outside the @hydra.main scripts (used by the notebook).
    Uses an absolute config dir (default <cwd>/conf) so it works under Jupyter/nbconvert, and
    clears any prior Hydra global state so it can be called repeatedly in one process."""
    import os

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = os.path.abspath(config_dir or os.path.join(os.getcwd(), "conf"))
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="config", overrides=overrides or [])


def to_device(obs, device):
    if isinstance(obs, dict):
        return {k: v.to(device) for k, v in obs.items()}
    return obs.to(device)


def expand_seeds(seeds, n_episodes):
    """Deterministically expand a base seed list to exactly n_episodes seeds."""
    seeds = list(seeds)
    out = []
    k = 0
    while len(out) < n_episodes:
        out.append(int(seeds[k % len(seeds)]) + (k // len(seeds)) * 100003)
        k += 1
    return out[:n_episodes]


@torch.no_grad()
def rollout_metrics(env, agent, device, n_episodes, seeds, max_steps, deterministic=True):
    """Run n_episodes deterministically (given the seed list) and aggregate the §9.1
    metrics. Episodes run in parallel batches of the env's num_envs; each episode runs the
    full horizon and the end-of-episode geometric state is read from env.evaluate()."""
    base = env.unwrapped
    nb = base.num_envs
    all_seeds = expand_seeds(seeds, n_episodes)
    tot_sorted = tot_mis = tot_parcels = 0.0
    n_all_placed = 0
    steps_sum = 0.0
    counted = 0
    for start in range(0, n_episodes, nb):
        batch_seeds = all_seeds[start:start + nb]
        take = len(batch_seeds)
        if take < nb:  # pad final batch; extras are discarded
            batch_seeds = batch_seeds + all_seeds[: nb - take]
        obs, _ = env.reset(seed=batch_seeds)
        obs = to_device(obs, device)
        for _ in range(max_steps - 1):
            obs, _, _, _, _ = env.step(agent.act(obs, deterministic=deterministic))
            obs = to_device(obs, device)
        ev = base.evaluate()
        sc = ev["success_count"][:take]
        tot_sorted += sc.sum().item()
        tot_mis += ev["mis_sort_count"][:take].sum().item()
        tot_parcels += base.num_parcels * take
        n_all_placed += ev["all_placed"][:take].sum().item()
        steps_sum += ev["steps_to_complete"][:take].float().sum().item()
        counted += take
    return dict(
        n_episodes=counted,
        num_parcels=base.num_parcels,
        sort_accuracy=tot_sorted / max(tot_parcels, 1),
        mean_sorted=tot_sorted / max(counted, 1),
        all_placed_rate=n_all_placed / max(counted, 1),
        mean_steps=steps_sum / max(counted, 1),
        mis_sort_rate=tot_mis / max(tot_parcels, 1),
    )


def print_metrics(role, difficulty, obs_mode, m, hard=False):
    """Print the §9.1 summary block."""
    print("-" * 50)
    print(f"{role}  difficulty={difficulty}  n_episodes={m['n_episodes']}  obs_mode={obs_mode}")
    print(f"  sort_accuracy:        {m['sort_accuracy']:.3f}        # PRIMARY")
    print(f"  mean_sorted/episode:  {m['mean_sorted']:.2f} / {m['num_parcels']}")
    print(f"  all_placed_rate:      {m['all_placed_rate']:.3f}")
    if hard:
        print(f"  mean_steps:           {m['mean_steps']:.0f}         # speed (hard tiebreaker)")
    print(f"  mis_sort_rate:        {m['mis_sort_rate']:.3f}        # diagnostic")
    print("-" * 50, flush=True)


def load_agent(ckpt_path, env, device, entrypoint=None):
    """Load the policy that eval.py/test.py will drive.

    The policy only ever needs to satisfy the contract in policy.py: ``act(obs, deterministic)``
    taking the locked-mode observation and returning actions in [-1, 1]. It is built either by:

    * a custom ENTRYPOINT ``"module:function"`` (config ``policy=...``) where
      ``function(checkpoint, sample_obs, action_space, device) -> policy`` returns ANY object
      satisfying the contract (RL net, scripted controller, CV pipeline, ...); or
    * (default) the built-in ``Agent``, rebuilt from a sample obs and the checkpoint weights, so
      a train.py checkpoint runs with no code changes (BUILD_SPEC §9).

    Returns (policy, checkpoint_dict_or_None).
    """
    sample_obs = to_device(env.reset(seed=0)[0], device)
    action_space = env.single_action_space

    if entrypoint:
        import importlib

        mod_name, fn_name = entrypoint.split(":")
        fn = getattr(importlib.import_module(mod_name), fn_name)
        policy = fn(ckpt_path, sample_obs, action_space, device)
        assert hasattr(policy, "act"), f"policy from {entrypoint} must define .act(obs, deterministic=True)"
        return policy, None

    from warehouse_sort.policy import Agent

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent = Agent(sample_obs, action_space.shape[0]).to(device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()
    return agent, ckpt


def record_eval_video(cfg, obs_mode, randomization, agent, device, out_dir,
                      n_envs=4, seed=0, max_steps=None):
    """Record a short policy rollout to an mp4 using ManiSkill's RecordEpisode wrapper, with
    render_mode='all' so the saved video shows ALL views tiled: the human render camera AND the
    policy's sensor camera input (the Panda wrist camera). Called by eval.py on every run."""
    from mani_skill.utils.wrappers.record import RecordEpisode

    env = _gym_make(cfg, obs_mode, randomization, n_envs, render_mode="all")
    if obs_mode == "rgb":
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    env = RecordEpisode(
        env, output_dir=out_dir, save_trajectory=False, save_video=True,
        video_fps=20, max_steps_per_video=cfg.max_episode_steps,
    )
    obs, _ = env.reset(seed=seed)
    steps = max_steps or cfg.max_episode_steps
    for _ in range(steps):
        obs, _, _, _, _ = env.step(agent.act(to_device(obs, device), deterministic=True))
    env.close()  # flushes the mp4
    return out_dir


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def log_run_header(cfg, role: str):
    """Log the resolved config + git hash at the start of every run (BUILD_SPEC §8)."""
    print("=" * 70)
    print(f"[{role}] git={git_hash()}")
    print("-" * 70)
    print(OmegaConf.to_yaml(cfg, resolve=True).rstrip())
    print("=" * 70, flush=True)


def make_env(
    cfg,
    obs_mode: str,
    randomization: dict,
    num_envs: Optional[int] = None,
    render_mode: Optional[str] = None,
    record_metrics: bool = True,
    ignore_terminations: bool = True,
    video_dir: Optional[str] = None,
):
    """Construct the WarehouseSort env + standard ManiSkill vector wrappers.

    Returns (vector_env, is_rgb). For rgb obs the observation is a dict {"rgb", "state"};
    for state obs it is a flat tensor.

    ``ignore_terminations`` controls auto-reset behaviour (ManiSkill's `partial_reset`):
    train.py passes ``False`` so each env resets the instant it succeeds (matching the
    ManiSkill PPO baseline); the periodic eval / test.py / eval.py keep the default ``True``
    so every episode runs the full fixed horizon and the end-of-episode geometric state is
    read once.

    ``video_dir`` (set for the in-loop eval, mirroring the baseline's RecordEpisode on
    eval_envs): if given, the env renders ``all`` views (render camera + wrist sensor) and a
    RecordEpisode wrapper saves an mp4 of each eval rollout there.
    """
    from mani_skill.utils.wrappers.record import RecordEpisode

    n = int(num_envs if num_envs is not None else cfg.num_envs)
    is_rgb = obs_mode == "rgb"
    if video_dir is not None and render_mode is None:
        render_mode = "all"          # render camera + wrist sensor, tiled (see env.render)
    env = _gym_make(cfg, obs_mode, randomization, n, render_mode)
    if is_rgb:
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if video_dir is not None:
        env = RecordEpisode(
            env, output_dir=video_dir, save_trajectory=False, save_video=True,
            video_fps=20, max_steps_per_video=cfg.max_episode_steps,
        )
    env = ManiSkillVectorEnv(
        env, num_envs=n, ignore_terminations=ignore_terminations, record_metrics=record_metrics
    )
    return env, is_rgb


def index_obs(obs, mask):
    """Index a (possibly dict) observation along the env/batch axis. Used to bootstrap the
    value of the `final_observation` of just the envs that terminated this step."""
    if isinstance(obs, dict):
        return {k: v[mask] for k, v in obs.items()}
    return obs[mask]
