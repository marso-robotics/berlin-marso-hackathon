"""Generate an imitation-learning demonstration dataset for WarehouseSort-v1 (easy).

This is the IL track's data step. It rolls out the **scripted waypoint policy**
(``examples/scripted_policy.py`` — we reuse its grasp/place logic, we do not reinvent it)
across many seeds on the chosen difficulty and records the
episodes in ManiSkill's standard trajectory format (``.h5`` + ``.json``) via the official
``RecordEpisode`` wrapper.

After recording the raw demos this script invokes ManiSkill's own ``replay_trajectory`` tool
to produce the training-ready dataset(s) the learning scripts expect:

  * **state**  : ``--obs-mode state``  (the easy-milestone BC input)
  * **rgb**    : ``--obs-mode rgb``     (scene-cam, for the RGB Diffusion Policy)

We follow ManiSkill's pipeline rather than hand-rolling the dataset: record -> replay ->
train. See ``il/README.md``.

Usage:
  pixi run python il/gen_demos.py --num-episodes 60
  pixi run python il/gen_demos.py --num-episodes 60 --no-replay   # just record raw demos
"""

import argparse
import os
import subprocess
import sys

import gymnasium as gym
import numpy as np

# reuse the scripted policy (grasp/place logic) and its per-difficulty env kwargs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warehouse_sort  # noqa: F401  (registers WarehouseSort-v1)
from examples.scripted_policy import DIFFICULTY_KWARGS, scripted_episode
from mani_skill.utils.wrappers.record import RecordEpisode

CONTROL_MODE = "pd_ee_delta_pos"   # the fixed task controller (README §Action space)


def record_raw_demos(out_dir, difficulty, num_episodes, action_noise, base_seed, max_steps,
                     obs_camera="scene", return_home=False, home_hold=5):
    """Roll the scripted policy for ``num_episodes`` seeds and save raw .h5 + .json.

    ``obs_camera`` is baked into the recorded env_kwargs, so the later ``replay_trajectory`` step
    re-renders the rgb obs from that same camera ("scene" -> fixed third-person view)."""
    os.makedirs(out_dir, exist_ok=True)
    kwargs = DIFFICULTY_KWARGS[difficulty]
    n_parcels = kwargs["num_parcels"]

    env = gym.make(
        "WarehouseSort-v1",
        num_envs=1,
        obs_mode="state",
        control_mode=CONTROL_MODE,
        sim_backend="gpu",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
        obs_camera=obs_camera,
        **kwargs,
    )
    # RecordEpisode saves one trajectory per episode (flushed on each reset / on close) into a
    # single .h5 + .json pair, recording env states too so replay_trajectory can re-render obs.
    env = RecordEpisode(
        env,
        output_dir=out_dir,
        trajectory_name="trajectory",
        save_trajectory=True,
        save_video=False,
        record_env_state=True,
        save_on_reset=True,
    )

    n_success, kept = 0, 0
    for i in range(num_episodes):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        history = scripted_episode(
            env, max_steps=max_steps, seed=seed, action_noise=action_noise, rng=rng,
            stop_on_success=False, return_home=return_home, home_hold=home_hold,
        )
        info = history[-1][-1]
        sc = info.get("success_count")
        sc_val = float(sc.item() if hasattr(sc, "item") else sc)
        success = bool(info.get("success")[0].item()) if hasattr(info.get("success"), "item") \
            else bool(info.get("success"))
        n_success += int(success)
        kept += 1
        print(f"  ep {i:3d} seed={seed} sorted={sc_val:.0f}/{n_parcels} success={success}",
              flush=True)
    env.close()  # final flush

    h5 = os.path.join(out_dir, "trajectory.h5")
    print(f"\nrecorded {kept} episodes  ({n_success} full-success)  -> {h5}", flush=True)
    return h5


def replay(h5_path, obs_mode, suffix):
    """Run ManiSkill's replay_trajectory to produce a training-ready dataset in `obs_mode`.

    Records the same actions (pd_ee_delta_pos) with freshly rendered observations in the target
    obs mode, on the GPU backend (no control-mode conversion -> GPU is supported)."""
    out = h5_path.replace(".h5", f".{suffix}.h5")
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "_replay.py"),
        "--traj-path", h5_path,
        "--obs-mode", obs_mode,
        "--sim-backend", "physx_cuda",
        "--save-traj",
        "--num-envs", "1",
    ]
    print(f"\n[replay -> {obs_mode}]  {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    # replay writes <name>.<obs_mode>.<backend>.h5 next to the source; report what landed
    print(f"[replay] done ({obs_mode}); see {os.path.dirname(h5_path)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    ap.add_argument("--num-episodes", type=int, default=60)
    ap.add_argument("--action-noise", type=float, default=0.03,
                    help="std of Gaussian noise on xyz action dims during collection; spreads "
                         "the demo state distribution to fight BC covariate shift (0 = none)")
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=200,
                    help="per-episode cap; demos end shortly after the last placement settles")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-replay", action="store_true",
                    help="only record raw demos; skip the replay_trajectory obs conversion")
    ap.add_argument("--obs-modes", nargs="*", default=["state", "rgb"],
                    help="which obs representations to produce via replay_trajectory")
    ap.add_argument("--obs-camera", default="scene", choices=["scene"],
                    help="image obs camera (scene = fixed third-person; the only supported camera)")
    ap.add_argument("--return-home", action="store_true",
                    help="after the last placement, return the arm home (clearer videos but adds "
                         "~30 idle frames the learner over-weights). Default off -> sharp demos.")
    ap.add_argument("--home-hold", type=int, default=5,
                    help="frames to hold still at the end (lets the box settle so success latches)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "demos", args.difficulty
    )
    h5 = record_raw_demos(out_dir, args.difficulty, args.num_episodes,
                          args.action_noise, args.base_seed, args.max_steps,
                          obs_camera=args.obs_camera, return_home=args.return_home,
                          home_hold=args.home_hold)

    if not args.no_replay:
        for om in args.obs_modes:
            replay(h5, om, om)


if __name__ == "__main__":
    main()
