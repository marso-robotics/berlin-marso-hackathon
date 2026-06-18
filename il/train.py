"""Thin runner over the vendored ManiSkill IL baselines for WarehouseSort-v1 (easy).

This does NOT reimplement any learning logic — it just invokes the real vendored baseline
scripts (``il/baselines/{bc,diffusion_policy,act}``) with the demo paths, control mode, and
the eval horizon this task needs, so competitors have one consistent entrypoint.

  pixi run python il/train.py dp                 # Diffusion Policy, state  (recommended)
  pixi run python il/train.py bc                 # MLP behaviour cloning, state
  pixi run python il/train.py bc_rgb             # RGB (wrist-cam) BC
  pixi run python il/train.py act                # ACT, state
  pixi run python il/train.py dp --total-iters 30000 --exp-name my_run   # extra args pass through

Outputs (checkpoints + tensorboard) land under ``il/baselines/<method>/runs/<exp-name>/``.
Evaluate the resulting checkpoint through the project's judge harness, e.g.:

  pixi run python eval.py difficulty=easy obs_mode=state max_episode_steps=200 \
      policy=warehouse_sort.il_policy:load_dp \
      checkpoint=il/baselines/diffusion_policy/runs/<exp>/checkpoints/best_eval_success_at_end.pt \
      eval_config=conf/eval/default.yaml
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.join(HERE, "demos", "easy")
STATE_H5 = os.path.join(DEMOS, "trajectory.state.pd_ee_delta_pos.physx_cuda.h5")
RGB_H5 = os.path.join(DEMOS, "trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5")

# method -> (baseline cwd, script, demo path, default args). Horizon 180 leaves room for the
# full pick-place-of-both-parcels + clean return-home that the demos contain.
COMMON = ["--env-id", "WarehouseSort-v1", "--control-mode", "pd_ee_delta_pos",
          "--sim-backend", "gpu", "--max-episode-steps", "200"]
METHODS = {
    "dp": dict(dir="diffusion_policy", script="train.py", demo=STATE_H5,
               args=["--total-iters", "30000", "--batch-size", "256",
                     "--obs-horizon", "2", "--act-horizon", "8", "--pred-horizon", "16",
                     "--num-eval-envs", "16", "--num-eval-episodes", "32",
                     "--eval-freq", "5000", "--log-freq", "1000", "--save-freq", "10000",
                     "--no-capture-video", "--exp-name", "warehouse_state_dp"]),
    # RGB (scene-cam) Diffusion Policy: image + robot proprioception only (NO privileged state),
    # rgb-only (no depth). Fixed image input shape -> the SAME policy can run across difficulties.
    # --capture-video records an mp4 of the eval rollout each eval_freq under runs/<exp>/videos/.
    "dp_rgb": dict(dir="diffusion_policy", script="train_rgbd.py", demo=RGB_H5,
                   args=["--obs-mode", "rgb", "--obs-camera", "scene",
                         "--total-iters", "30000", "--batch-size", "128",
                         "--obs-horizon", "2", "--act-horizon", "8", "--pred-horizon", "16",
                         "--num-eval-envs", "4", "--num-eval-episodes", "8",
                         "--eval-freq", "5000", "--log-freq", "1000", "--save-freq", "10000",
                         "--capture-video", "--exp-name", "warehouse_rgb_dp"]),
    "bc": dict(dir="bc", script="bc.py", demo=STATE_H5,
               args=["--total-iters", "20000", "--batch-size", "512", "--lr", "3e-4",
                     "--num-eval-envs", "16", "--num-eval-episodes", "32",
                     "--eval-freq", "5000", "--log-freq", "2000", "--save-freq", "5000",
                     "--no-capture-video", "--exp-name", "warehouse_state_bc"]),
    "bc_rgb": dict(dir="bc", script="bc_rgb.py", demo=RGB_H5,
                   args=["--total-iters", "30000", "--batch-size", "256",
                         "--num-eval-envs", "8", "--num-eval-episodes", "16",
                         "--eval-freq", "5000", "--log-freq", "1000", "--save-freq", "10000",
                         "--no-capture-video", "--exp-name", "warehouse_rgb_bc"]),
    "act": dict(dir="act", script="train.py", demo=STATE_H5,
                args=["--total-iters", "30000", "--batch-size", "128", "--num-queries", "16",
                      "--no-temporal-agg",
                      "--num-eval-envs", "8", "--num-eval-episodes", "16",
                      "--eval-freq", "5000", "--log-freq", "1000", "--save-freq", "10000",
                      "--no-capture-video", "--exp-name", "warehouse_state_act"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=list(METHODS))
    ap.add_argument("--demo-path", default=None, help="override the demo .h5")
    args, extra = ap.parse_known_args()

    m = METHODS[args.method]
    demo = args.demo_path or m["demo"]
    if not os.path.exists(demo):
        sys.exit(f"demo dataset not found: {demo}\nrun:  pixi run python il/gen_demos.py")
    cwd = os.path.join(HERE, "baselines", m["dir"])
    cmd = [sys.executable, m["script"], "--demo-path", demo] + COMMON + m["args"] + extra
    print(f"[il/train] cwd={cwd}\n[il/train] {' '.join(cmd)}", flush=True)
    sys.exit(subprocess.run(cmd, cwd=cwd).returncode)


if __name__ == "__main__":
    main()
