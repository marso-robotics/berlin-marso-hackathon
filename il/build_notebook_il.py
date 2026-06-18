"""Generates il/notebook_il.ipynb — the IMITATION-LEARNING track notebook (Colab-friendly).

Like the RL notebook, this is a THIN RUNNER over the real code: every cell calls the actual
scripts (`il/gen_demos.py`, `il/train.py`, `eval.py`), the vendored ManiSkill baselines, and
the `warehouse_sort` package. It never reimplements env/training logic and never references
`judge/`.  Run:  python il/build_notebook_il.py
"""
import json
import os

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# WarehouseSort-v1 — Imitation-Learning track (easy)

Clone the **scripted** waypoint policy into a learned one, the ManiSkill way:
**record demos → `replay_trajectory` into the training representation → train a vendored
ManiSkill baseline → evaluate through the same `eval.py` the judges use.**

We work on **easy** (state obs, 2 parcels, fixed poses). The recommended method is
**Diffusion Policy** (state): its action chunking avoids the compounding-error blow-up that
sinks a plain MLP behaviour-cloner on this fixed single-path dataset.

> **Select a GPU runtime:** *Runtime → Change runtime type → GPU.* ManiSkill 3 needs CUDA.
""")

md("## 1. Install")
code(r"""
# Colab: clone then install. (Skip the clone if already inside the repo.)
# !git clone <YOUR_REPO_URL> warehouse_sort && cd warehouse_sort
!pip install -q "mani_skill>=3.0.0" "hydra-core>=1.3" "omegaconf>=2.3"
!pip install -q -e .
# IL baselines need these extra deps (DP/ACT use diffusers; ACT also needs torchvision):
!pip install -q diffusers torchvision
print("install done")
""")

md(r"""
## 2. Headless rendering (Colab)

ManiSkill/SAPIEN render offscreen via Vulkan, which works on Colab GPU runtimes. Import the
package (registers `WarehouseSort-v1`) and confirm the GPU.
""")
code(r"""
import torch, warehouse_sort  # registers the env
print("cuda:", torch.cuda.is_available())
""")

md(r"""
## 3. Generate demonstrations

Roll the scripted policy across 60 seeds on easy, record ManiSkill `.h5`+`.json`, then replay
into **state** and **wrist-cam rgb** datasets. Each demo ends with a clean finish (release →
lift off → return home). Small demo-time action noise widens the state distribution to fight
behaviour-cloning covariate shift.
""")
code(r"""
!python il/gen_demos.py --num-episodes 60 --action-noise 0.05
import glob; print(glob.glob("il/demos/easy/*.h5"))
""")

md("## 4. Inspect the demos (documented shapes)")
code(r"""
import h5py, numpy as np
sd = h5py.File("il/demos/easy/trajectory.state.pd_ee_delta_pos.physx_cuda.h5","r")
t = sd["traj_0"]
print("state obs:", t["obs"].shape, t["obs"].dtype, "| actions:", t["actions"].shape)  # (T,54), (T-1,4)
rd = h5py.File("il/demos/easy/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5","r")
rgb = rd["traj_0"]["obs"]["sensor_data"]["hand_camera"]["rgb"]
print("wrist rgb:", rgb.shape, rgb.dtype)  # (T,128,128,3) uint8
import matplotlib.pyplot as plt
plt.imshow(rgb[len(rgb)//2]); plt.title("wrist camera (mid-episode)"); plt.axis("off"); plt.show()
""")

md(r"""
## 5. Train — Diffusion Policy (state)

`il/train.py` is a thin wrapper over the vendored `diffusion_policy/train.py` with the demo
path, control mode and 180-step eval horizon wired in. Small `--total-iters` here so it
finishes on a Colab GPU; scale up for a stronger policy.
""")
code(r"""
!python il/train.py dp --total-iters 8000 --eval-freq 4000 --num-eval-envs 8 --num-eval-episodes 16
""")

md(r"""
## 6. Evaluate through the judges' `eval.py`

The trained checkpoint is loaded via the policy entrypoint `warehouse_sort.il_policy:load_dp`
(satisfies the `act(obs) -> action` contract), so it runs through the **same `eval.py`**.
""")
code(r"""
import glob
ckpt = sorted(glob.glob("il/baselines/diffusion_policy/runs/warehouse_state_dp/checkpoints/best_eval_success_*.pt"))[-1]
!python eval.py difficulty=easy obs_mode=state max_episode_steps=180 \
    policy=warehouse_sort.il_policy:load_dp checkpoint={ckpt} \
    eval_config=conf/eval/default.yaml num_envs=32
""")

md(r"""
## 7. RGB Diffusion Policy (scene camera) — image + proprioception only

The state policy is parcel-count-specific. The **rgb** policy sees a fixed third-person
**scene camera** (whole table/bins/robot always visible) plus robot proprioception — **no
privileged state, no depth** — so the same checkpoint is parcel-count-agnostic and can be run
across difficulties. (The wrist camera fails here: a grasped parcel occludes it.)
""")
code(r"""
# Train RGB DP on the scene-camera demos (small iters here; scale up for a stronger policy).
!python il/train.py dp_rgb --total-iters 8000 --eval-freq 4000 --num-eval-envs 8 --num-eval-episodes 16
""")
code(r"""
# Eval the rgb DP through the judges' eval.py (obs_camera=scene). load_dp_rgb is closed-loop.
import glob
ckpt = sorted(glob.glob("il/baselines/diffusion_policy/runs/warehouse_rgb_dp/checkpoints/best_eval_success_*.pt"))[-1]
!python eval.py difficulty=easy obs_mode=rgb obs_camera=scene \
    policy=warehouse_sort.il_policy:load_dp_rgb checkpoint={ckpt} \
    eval_config=conf/eval/default.yaml num_envs=16
""")

md(r"""
## 8. Watch a rollout

`eval.py` saves a rollout mp4 (render + sensor views) under its Hydra output dir each run.
""")
code(r"""
import glob, os
from IPython.display import Video
vids = sorted(glob.glob("outputs/**/videos/*.mp4", recursive=True), key=os.path.getmtime)
print(vids[-1]); Video(vids[-1], embed=True, width=640)
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "notebook_il.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
