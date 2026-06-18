# Warehouse Colour-Sort Challenge

Train a Franka Panda arm to pick parcels from the inbound zone and place each into the output bin
that matches the colour of the **tag on top of the parcel**. Your score is the number of parcels
sorted into the correct bin.

You are given the environment, observations, actions, and success condition. **Reward shaping is up
to you — that's the core of your submission.** A sparse reward is the default, and a simple worked
example (state-based) is included to learn from; the teams that design the best reward (and policy)
win.

---

## The task

- A Franka Panda arm with a parallel gripper is mounted at a tabletop workstation.
- **Parcels** look like brown warehouse cardboard boxes. Each carries a **coloured rectangular tag**
  (a sticker/label) on its **top face** — red or blue. The tag, not the box
  colour, tells you where the parcel goes.
- Parcels spawn in the **inbound zone** in the centre of the table, in front of the robot.
- There is one **colour-coded output bin per tag colour** — a red bin, a blue bin, etc. Bins are
  **low-walled and wide**, placed to the **left and right** of the robot.
- **Goal:** place each parcel into the bin whose colour matches the parcel's tag.
- **Score:** number of correctly-sorted parcels per episode.

### Colour → bin mapping

Tag colour → matching bin colour (red tag → red bin, blue tag → blue bin). The mapping is by **bin
colour, not by side** — at hard difficulty the bins' left/right positions can swap between episodes,
so you must identify the bins by colour, not location.

---

## Difficulty levels

Switch with `difficulty=easy|medium|hard`.

| Level  | Parcels | Layout                 | Tag colours | Observation default |
|--------|---------|------------------------|-------------|---------------------|
| easy   | 2       | fixed poses            | 2           | **state** (plumbing check) |
| medium | 4       | randomised poses, tag always top-visible | 2 | **rgb** (wrist cam) |
| hard   | 6       | randomised + light clutter, tag always top-visible | 2 | **rgb** (wrist cam) + heavy randomisation |

**Easy** exists only to confirm everything runs — it hands you tag colours directly as state.
**Medium and hard are the real challenge** and are vision-based: your policy sees the world through
the Panda's wrist camera and must detect each parcel's tag and the bin colours.

**Hard adds, on top of medium:**
- **Bin positions swap** between episodes (the red bin may be on the left or the right) — you must
  identify bins by colour, not memorise a side.
- **Background variation:** table and floor colours change.
- **Lighting variation.**
- **Appearance variation:** slightly different cardboard shades and tag-colour shades.
- A **secondary speed metric** (how fast you sort), used as a tiebreaker — correct-sort count is
  still the primary score.

---

## Observations

Selectable with `obs_mode=state|rgb`. Default is set per difficulty (above).

### `rgb` (default for medium/hard) — Panda wrist camera
- Source camera: the Panda wrist camera, uid **`hand_camera`** (the `panda_wristcam` robot's
  RealSense-style gripper camera).
- Image tensor at `obs["sensor_data"]["hand_camera"]["rgb"]`, shape `(num_envs, H, W, 3)`,
  **dtype `uint8`, value range `[0, 255]`, channel order `RGB`** (ManiSkill convention), on the
  GPU device. Default `H = W = 128` (set via `camera.width` / `camera.height`).
- Proprioception is included alongside the image (standard ManiSkill `rgb` obs). If you wrap
  with ManiSkill's `FlattenRGBDObservationWrapper` (as the starter scripts do), you get a dict
  `{"rgb": (num_envs,H,W,3) uint8, "state": (num_envs, 26) float32}` where the **26-dim
  `state`** is, in order: `agent.qpos` (9) · `agent.qvel` (9) · `tcp_pose` (7 = xyz + quat
  `wxyz`) · `is_grasped` (1). No privileged parcel/bin info is exposed in `rgb` mode.

### `state` (default for easy) — privileged low-dim vector
- Flat `float32` tensor, shape `(num_envs, 54)` for easy (2 parcels). The flatten order is:

  | slice | field | dims | meaning |
  |-------|-------|------|---------|
  | `[0:9]`   | `agent.qpos`       | 9 | joint positions (7 arm + 2 gripper) |
  | `[9:18]`  | `agent.qvel`       | 9 | joint velocities |
  | `[18:25]` | `tcp_pose`         | 7 | gripper TCP pose: xyz + quat `wxyz` |
  | `[25:26]` | `is_grasped`       | 1 | 1.0 if any parcel is grasped |
  | `[26:40]` | `parcel_pose`      | `P×7` | per-parcel pose (xyz + quat `wxyz`), P=num_parcels |
  | `[40:44]` | `parcel_tag`       | `P×2` | per-parcel **tag colour** one-hot `[red, blue]` |
  | `[44:50]` | `bin_position`     | `2×3` | xyz of bin colour 0 (red) then colour 1 (blue) |
  | `[50:54]` | `bin_color`        | `2×2` | bin colour one-hot (identity matrix; index = colour id) |

  For other parcel counts the `parcel_*` slices scale by `P`; total dim
  `= 26 + P*7 + P*2 + 6 + 4`. Parcels are ordered consistently; parcel `j`'s tag is
  `parcel_tag[2j:2j+2]`, and its correct destination is the bin with the matching colour id.

You may use either mode at any difficulty, but note: medium/hard are designed to be solved from
`rgb`, and **judging locks the obs mode to the difficulty default** (state for easy, rgb for
medium/hard) — `eval.py` ignores any `obs_mode` override, so you cannot be judged on privileged
`state` at medium/hard.

---

## Action space

Fixed `pd_ee_delta_pos` controller, **4 continuous dims, all normalised to `[-1, 1]`**:

| dims | meaning |
|------|---------|
| `[0:3]` | end-effector **delta position** (x, y, z), scaled to ±0.1 m per step (gripper held pointing down) |
| `[3]`   | **gripper**: `+1` = open, `-1` = close |

The action space is fixed — you do not redesign it.

---

## Reward (yours to build)

The starter ships a **sparse** reward: `+1` when a parcel lands in its correct-colour bin, `0`
otherwise. This will train, but slowly.

A **simple example dense reward** is also included (`reward=example_dense`) — a readable, normalised,
state-based shaping in the style of ManiSkill's `PickCube` (reach → grasp → move to correct bin →
place). It is deliberately the simplest version that works and is **not** optimised — it's there to
show a working pattern, not to be a strong baseline. **Replace or improve it with your own reward.**
This is where the competition is won, especially for the vision levels.

You are free to choose your algorithm, network architecture, visual encoder, and reward. You may not
change the environment, action space, observation interface, or success condition.

---

## Bring your own policy (not just RL, not a fixed architecture)

You are **not** locked into the starter's network or even into RL. The only thing `eval.py`
requires is an object satisfying this contract:

```python
policy.act(obs, deterministic=True) -> Tensor   # shape (num_envs, action_dim), values in [-1, 1]
```

`obs` is **exactly** the environment's observation in the difficulty's locked mode (a flat state
tensor for easy; a `{"rgb", "state"}` dict for medium/hard) and nothing else — the policy never
receives the env, the ground-truth state, or the scorer, so it can't read privileged info or game
the geometric success check. Any RL net, scripted controller, classical CV + control pipeline, or
behaviour-cloning model that meets the contract works.

Point `eval.py`/`test.py` at your policy with a loader entrypoint:

```bash
pixi run python eval.py difficulty=hard checkpoint=<path> eval_config=conf/eval/default.yaml \
    policy=my_submission:load_policy
```

```python
# my_submission.py
def load_policy(checkpoint, sample_obs, action_space, device):
    # build/restore anything; return an object with .act(obs, deterministic=True)
    return MyPolicy(checkpoint, sample_obs, action_space, device)
```

If you leave `policy` unset, the built-in `Agent` is loaded from your checkpoint (so a `train.py`
checkpoint runs unchanged). Judging uses the same `policy=` entrypoint, so your architecture choice
is entirely yours.

---

## Scoring

**Primary metric: sort accuracy** — fraction of parcels placed in the correct-colour bin, averaged
over the eval episodes. A parcel is correctly sorted when it rests inside the bin whose colour matches
its **tag colour**. The exact geometric check, per parcel `j` with tag colour id `t` and the bin of
colour `t` centred at `(bx, by)`:

```
|parcel.x - bx| < 0.11 m   AND   |parcel.y - by| < 0.13 m   AND   0 < parcel.z < 0.06 m
```

i.e. the parcel body is inside the bin's 0.22 m × 0.26 m footprint and settled low in the bin
(below 0.06 m, under the low ~0.05 m wall rim).
Scoring is geometric and deterministic — it does not depend on a camera read, though the scene is
built so correct sorting is also visually obvious from the top camera. A parcel in the *wrong*-colour
bin counts as a mis-sort (diagnostic), not a success, and is not retried.

`eval.py` reports:
- `sort_accuracy` — **primary ranking metric**
- `mean_sorted/episode` — parcels sorted out of total
- `all_placed_rate` — fraction of episodes where every parcel was placed (anywhere)
- `mean_steps` — **speed, hard-level tiebreaker only** (lower is better)
- `mis_sort_rate` — wrong-bin placements (diagnostic)

The episode ends when all parcels are placed or the step limit is reached. At **hard**, speed
(`mean_steps`) breaks ties between policies with equal sort accuracy.

---

## Quickstart (notebook)

If you'd rather start in a notebook, open **`notebook.ipynb`** (works in Google Colab — select a GPU
runtime). It walks through the whole pipeline step by step: install, build the env, view the parcels
and bins, inspect the wrist-camera and state observations, run a short training, then test and eval,
finishing with a rendered rollout of the trained policy. The notebook calls the same scripts and
configs documented below, so anything you learn there maps directly onto the command-line workflow.

The training cell uses small settings so it finishes quickly; scale up `total_steps` / `num_envs`
(marked in the notebook) for real runs.

---

## Setup

This repo is [pixi](https://pixi.sh)-managed; all commands run inside the pixi environment.

```bash
pixi install          # create the environment from pixi.toml / pixi.lock
pixi run install      # pip install -e .  (installs the warehouse_sort package + deps)
```

Then run any script through pixi, e.g. `pixi run python train.py difficulty=easy`. A CUDA GPU is
required (ManiSkill 3 GPU simulation). The convenience tasks `pixi run train|test|eval` are also
defined (append Hydra overrides, e.g. `pixi run train difficulty=medium`).

---

## Running

### Train
```bash
pixi run python train.py difficulty=medium
# override anything:
pixi run python train.py difficulty=medium num_envs=256 total_steps=5_000_000 seed=1
```
Checkpoints and the resolved config land in `outputs/<date>/<time>/`.

### Test (self-check, same distribution as training)
```bash
pixi run python test.py difficulty=medium checkpoint=outputs/<date>/<time>/ckpt.pt
```
Runs your policy on fresh same-distribution episodes (different seeds) and prints the score. Use this
to track progress.

### Eval (same interface as judging)
```bash
pixi run python eval.py difficulty=hard checkpoint=outputs/<date>/<time>/ckpt.pt eval_config=conf/eval/default.yaml
```
Reports the metrics above over N episodes. `conf/eval/default.yaml` is same-distribution as training,
so use it to rehearse the exact interface judges will use. The **held-out judging config is not
included in this repo** — judges run the same `eval.py` with your checkpoint against a held-out config
that widens and recombines the randomisation axes (see "How judging works" below). You won't see its
values; build for generalisation, not for the visible distribution.

---

## How judging works — read this

Judging uses **the exact same `eval.py` command and your trained checkpoint**, but with a
**held-out eval config** you do not have access to:

```bash
pixi run python eval.py difficulty=<level> checkpoint=<your_ckpt> eval_config=<held_out_config>
```

The held-out config keeps the **task identical** — same robot, same tag-colour → bin-colour mapping,
same success condition — but **widens and recombines the same randomisation you saw in training**.
At hard, training already varies bin sides, poses, lighting, backgrounds, and cardboard/tag shades
over *narrow* ranges; the held-out config widens those (full bin swaps, larger offsets, unseen
colours/lighting/shades) and combines them in ways not seen together in training.

The point: you can't pass by memorising a fixed layout (bins already move in training, so memorising
a side never works), and you can't pass by ignoring an axis (every axis is in training, so it's
learnable). You pass by genuinely learning to **read the tag, find the matching-colour bin, and place
there** — and having that skill hold up under conditions you haven't seen. That's true
generalisation.

**Implication:** a policy that memorises the visible training layout will score poorly. Build a
policy that *generalises* — that genuinely perceives and sorts under conditions it hasn't seen. The
visible `conf/eval/default.yaml` is same-distribution so you can rehearse the interface; your real
score comes from the held-out config.

Make sure your checkpoint loads and runs under `eval.py` with no code changes — that's exactly what
the judges will run.

---

## Kaggle submission

The competition runs on [Kaggle](https://www.kaggle.com/competitions/marso-hack) — your score is
posted to the leaderboard automatically when you submit a notebook.

### Step-by-step

1. **Upload your checkpoint** as a private Kaggle Dataset (kaggle.com/datasets/new → upload
   `ckpt.pt`).

2. **Copy the submission template notebook:**
   [marso-hack-submission-template](https://www.kaggle.com/code/albatllezfernndez/marso-hack-submission-template)
   → `Copy & Edit`.

3. **Add your dataset** to the notebook (+ Add Data → your checkpoint dataset).

4. **Set `CHECKPOINT_PATH`** in the first cell to your checkpoint's Kaggle path
   (`/kaggle/input/<your-dataset-slug>/ckpt.pt`).

5. **Run all cells.** The notebook installs dependencies, runs `eval.py` with
   `submission_csv=/kaggle/working/submission.csv`, and prints your `sort_accuracy`.

6. **Commit & submit** (top-right) — your score appears on the leaderboard.

> GPU runtime is required (ManiSkill 3 uses CUDA + Vulkan). Enable it under
> *Session options → Accelerator → GPU T4 × 1*.

### Local dry-run (same command the notebook uses)

```bash
pixi run python eval.py difficulty=hard checkpoint=<path> \
    eval_config=conf/eval/default.yaml \
    submission_csv=submission.csv
```

This writes `submission.csv` in the current directory alongside the normal metric output.

---

## Tips (not requirements)

- Get `difficulty=easy` (state) running first to confirm your training loop works, then move to
  `medium` (vision) — that's the real task.
- For vision, a frozen pretrained image encoder is a fast, robust starting point. From-scratch CNNs
  are slower to converge under time pressure.
- Render small and keep `num_envs` sized to your GPU — RGB observations are the main memory cost.
- Since judging is on a held-out distribution, anything that improves generalisation (domain
  randomisation, augmentation, robust encoders) is likely worth more than squeezing the training
  distribution.
