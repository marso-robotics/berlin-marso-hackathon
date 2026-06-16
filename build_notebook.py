"""Generates notebook.ipynb (BUILD_SPEC §12a). Run: python build_notebook.py
The notebook is a THIN RUNNER over the real package/scripts/configs -- it never duplicates
env, training, or reward logic, and never references judge/."""
import json

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Warehouse Colour-Sort Challenge — guided notebook

Train a Franka Panda to pick brown cardboard parcels and drop each into the bin that matches
the **colour of the tag on top of the parcel** (red tag → red bin, blue tag → blue bin).

This notebook is a **thin runner over the real code**: every cell calls the actual
`warehouse_sort` package, the `train.py` / `test.py` / `eval.py` scripts, and the Hydra configs
in `conf/`. It does not reimplement the environment, training, or reward.

It walks through: install → render each difficulty → inspect observations → a short training
run → test → eval → a rendered rollout of the trained policy.

> **Select a GPU runtime first:** *Runtime → Change runtime type → Hardware accelerator → GPU.*
> ManiSkill 3 requires a CUDA GPU.
""")

md(r"""
## 1. Install

In Google Colab, first get the repository, then install it. (If you are already running this
notebook from inside the repo, just run the install.)
""")

code(r"""
# --- Colab only: clone the repo, then cd into it ---
# !git clone <YOUR_REPO_URL> warehouse_sort && cd warehouse_sort
import os
# install ManiSkill 3 + this package (editable). torch/CUDA are preinstalled on Colab GPU runtimes.
!pip install -q "mani_skill>=3.0.0" "hydra-core>=1.3" "omegaconf>=2.3"
!pip install -q -e .
print("install done")
""")

md(r"""
## 2. Colab headless rendering

ManiSkill / SAPIEN render **offscreen via Vulkan**, which works on Colab GPU runtimes with no
virtual display. The cell below just verifies the GPU and that the env imports & registers.
""")

code(r"""
import torch
print("CUDA available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import numpy as np
import matplotlib.pyplot as plt
import warehouse_sort                       # registers the WarehouseSort-v1 env
from warehouse_sort.utils import make_env, load_agent, compose_cfg, rollout_metrics
print("warehouse_sort imported and env registered")
""")

md(r"""
## 3. Build the env and render each difficulty

We construct the real env at each difficulty through `make_env` (the same helper the scripts
use) and render the human camera. You should see brown parcels with **coloured top tags**, two
**colour-coded L/R bins**, and — at **hard** — visible randomisation (bin sides, background,
lighting) across the 4 parallel scenes.
""")

code(r"""
def render_difficulty(diff, n=4, seed=0):
    cfg = compose_cfg([f"difficulty={diff}"])
    env, _ = make_env(cfg, cfg.difficulty.obs_mode, cfg.randomization,
                      num_envs=n, render_mode="rgb_array")
    env.reset(seed=seed)
    frames = env.render().cpu().numpy()      # (n, H, W, 3) uint8
    env.close()
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(frames[i]); ax.axis("off"); ax.set_title(f"{diff} · scene {i}")
    plt.tight_layout(); plt.show()

for d in ["easy", "medium", "hard"]:
    render_difficulty(d)
""")

md(r"""
## 4. Inspect the observations

Two observation modes are exposed (config `obs_mode`); judging locks each difficulty to its
default — `state` for easy, `rgb` (Panda wrist camera) for medium/hard.

* **`rgb`** — wrist-camera image `(num_envs, H, W, 3)`, dtype `uint8`, RGB, range `[0,255]`
  (default 128×128), plus a proprioception `state` vector.
* **`state`** — privileged low-dim vector (parcel poses, parcel **tag** one-hot, bin positions
  **and bin colours**, robot proprioception). See the README for the exact layout.
""")

code(r"""
# --- rgb (wrist camera) at medium ---
cfg = compose_cfg(["difficulty=medium"])
env, _ = make_env(cfg, "rgb", cfg.randomization, num_envs=2)
obs, _ = env.reset(seed=0)
print("rgb obs is a dict:", {k: tuple(v.shape) for k, v in obs.items()}, "dtypes:",
      {k: str(v.dtype) for k, v in obs.items()})
wrist = obs["rgb"][0].cpu().numpy()          # (H, W, 3) uint8 — the wrist-camera view
env.close()
plt.figure(figsize=(4, 4)); plt.imshow(wrist); plt.axis("off")
plt.title("Panda wrist-camera RGB (medium)"); plt.show()

# --- state (privileged vector) at easy ---
cfg = compose_cfg(["difficulty=easy"])
env, _ = make_env(cfg, "state", cfg.randomization, num_envs=2)
obs, _ = env.reset(seed=0)
print("state obs tensor shape:", tuple(obs.shape), "dtype:", obs.dtype)
env.close()
""")

md(r"""
## 5. A short training run

We call the **real `train.py`** with small settings so it finishes quickly on a Colab GPU. This
is just to exercise the loop — **scale up `total_steps` and `num_envs` for real training** (this
task needs on the order of millions of environment steps, and a good reward, before it sorts
reliably; the bundled example reward is a starting point, not a strong baseline).

We use `reward=example_dense` (the provided state-based example shaping). The default reward is
`sparse`; designing a better reward is the core of your submission.
""")

code(r"""
# Short run -> checkpoint at outputs/nb_run/ckpt.pt. SCALE UP total_steps / num_envs for real training.
!python train.py difficulty=easy reward=example_dense total_steps=50000 num_envs=64 \
    ppo.ent_coef=0.01 hydra.run.dir=outputs/nb_run
print("checkpoint:", os.path.exists("outputs/nb_run/ckpt.pt"))
""")

md(r"""
## 6. Test (self-check, same distribution as training)

`test.py` runs the checkpoint on fresh same-distribution episodes and prints the §9.1 metrics.
""")

code(r"""
!python test.py difficulty=easy reward=example_dense checkpoint=outputs/nb_run/ckpt.pt
""")

md(r"""
## 7. Eval (the exact interface judges use)

`eval.py` is driven by an eval-config file. `conf/eval/default.yaml` is same-distribution as
training so you can rehearse the interface. Judging uses the **same command** with a held-out
config you don't see; the obs mode is locked to the difficulty default.
""")

code(r"""
!python eval.py difficulty=easy reward=example_dense checkpoint=outputs/nb_run/ckpt.pt \
    eval_config=conf/eval/default.yaml
""")

md(r"""
## 8. Render a trained-policy rollout

Finally we roll out the trained policy and render it (render camera + wrist camera side by
side, the same views the eval videos use), so the notebook ends on a visible result. (With only
the short training above the arm will reach/grasp but may not sort reliably yet — scale up
training to improve it.)
""")

code(r"""
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = compose_cfg(["difficulty=easy", "reward=example_dense"])
# render_mode="all" -> each frame is the render camera and the wrist camera side by side
env, _ = make_env(cfg, "state", cfg.randomization, num_envs=1, render_mode="all")
agent, _ = load_agent("outputs/nb_run/ckpt.pt", env, device)

obs, _ = env.reset(seed=0)
frames = []
for t in range(cfg.max_episode_steps):
    obs = obs.to(device) if not isinstance(obs, dict) else {k: v.to(device) for k, v in obs.items()}
    action = agent.act(obs, deterministic=True)
    obs, _, _, _, _ = env.step(action)
    frames.append(env.render()[0].cpu().numpy())
env.close()

# montage of the rollout over time
picks = np.linspace(0, len(frames) - 1, 8).astype(int)
fig, axes = plt.subplots(1, len(picks), figsize=(3 * len(picks), 3))
for ax, t in zip(axes, picks):
    ax.imshow(frames[t]); ax.axis("off"); ax.set_title(f"t={t}")
plt.tight_layout(); plt.show()

# optional: inline video if mediapy/imageio is available
try:
    import mediapy
    mediapy.show_video(frames, fps=20)
except Exception as e:
    print("(install mediapy for an inline video; montage shown above)", e)
""")

md(r"""
That's the full pipeline — all via the real package, scripts, and configs. Next steps:

* Move to `difficulty=medium` (vision via the wrist camera) — the real task.
* Replace the example reward with your own; that's where the competition is won.
* Scale up `total_steps` / `num_envs`, and build for **generalisation** (judging uses a
  held-out distribution that widens and recombines the training randomisation).
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "accelerator": "GPU", "colab": {"provenance": []},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

with open("notebook.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote notebook.ipynb with", len(cells), "cells")
