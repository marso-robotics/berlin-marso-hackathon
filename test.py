"""Self-check a checkpoint on SAME-DISTRIBUTION held-back episodes (same randomisation
ranges as training, different seeds) and print the §9.1 metrics. Use this to track progress.

  python test.py difficulty=easy checkpoint=outputs/<date>/<time>/ckpt.pt
  python test.py difficulty=hard checkpoint=<path> num_envs=64

Unlike eval.py, test.py lets you keep the obs mode you trained with (it does not lock it).
"""

import os

import hydra
import torch

from warehouse_sort.utils import (
    load_agent, log_run_header, make_env, print_metrics, record_eval_video, rollout_metrics,
)

# fixed self-check seeds (distinct from eval/heldout); same distribution as training
TEST_SEEDS = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    assert cfg.checkpoint, "pass checkpoint=<path to ckpt.pt>"
    log_run_header(cfg, "test")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    n_episodes = 16 if cfg.difficulty.name == "easy" else 32
    n_envs = min(cfg.num_envs, n_episodes)
    env, _ = make_env(cfg, cfg.obs_mode, cfg.randomization, num_envs=n_envs)
    agent, _ = load_agent(cfg.checkpoint, env, device, entrypoint=cfg.policy)

    m = rollout_metrics(env, agent, device, n_episodes, TEST_SEEDS, cfg.max_episode_steps)
    print_metrics("TEST", cfg.difficulty.name, cfg.obs_mode, m, hard=(cfg.difficulty.name == "hard"))
    env.close()

    # also save a rollout video (RecordEpisode, render + wrist views side by side)
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    vid_dir = os.path.join(out_dir, "videos")
    record_eval_video(cfg, cfg.obs_mode, cfg.randomization, agent, device, vid_dir,
                      n_envs=min(4, n_envs), seed=TEST_SEEDS[0])
    print(f"[test] saved rollout video (render + sensor views) -> {vid_dir}", flush=True)


if __name__ == "__main__":
    main()
