"""Record a single ACT-state rollout video (RecordEpisode) from a trained checkpoint.

Honours action chunking and lets you execute fewer steps than the prediction horizon:
  * --no-temporal-agg  -> receding horizon: run `num_queries` actions per prediction
    (recommended for delta-position control; temporal averaging dampens delta actions).
  * --temporal-agg     -> ACT's exponential temporal ensembling (query every step).
"""
import sys
from types import SimpleNamespace
import tyro
from dataclasses import dataclass

import torch
import warehouse_sort  # noqa
from act.make_env import make_eval_envs
from act.evaluate import evaluate
from train import Agent, Args  # vendored ACT


@dataclass
class Rec:
    ckpt: str = "runs/warehouse_state_act/checkpoints/best_eval_success_at_end.pt"
    video_dir: str = "/home/david/code/marso_hackathon/il/videos/state_act"
    num_queries: int = 30
    temporal_agg: bool = False
    max_episode_steps: int = 180
    episodes: int = 1


if __name__ == "__main__":
    rec = tyro.cli(Rec)
    device = "cuda"
    # full ACT Args defaults give the detr/transformer hyperparams the Agent needs
    args = Args(env_id="WarehouseSort-v1", control_mode="pd_ee_delta_pos",
                num_queries=rec.num_queries, max_episode_steps=rec.max_episode_steps,
                num_eval_envs=1, sim_backend="gpu")
    env_kwargs = dict(control_mode="pd_ee_delta_pos", reward_mode="sparse",
                      obs_mode="state", render_mode="all",
                      max_episode_steps=rec.max_episode_steps)
    envs = make_eval_envs("WarehouseSort-v1", 1, "gpu", env_kwargs, None, video_dir=rec.video_dir)
    agent = Agent(envs, args).to(device)
    ck = torch.load(rec.ckpt, map_location=device, weights_only=False)
    agent.load_state_dict(ck["ema_agent"])
    norm_stats = ck.get("norm_stats", None)
    eval_kwargs = dict(stats=norm_stats, num_queries=rec.num_queries,
                       temporal_agg=rec.temporal_agg, max_timesteps=rec.max_episode_steps,
                       device=device, sim_backend="gpu")
    m = evaluate(rec.episodes, agent, envs, eval_kwargs)
    import numpy as np
    print("success_at_end:", float(np.mean(m["success_at_end"])),
          " return:", float(np.mean(m["return"])))
    envs.close()
    print("video ->", rec.video_dir)
