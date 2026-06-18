# WarehouseSort — Color-Matching Pick-and-Place

A robotics generalization challenge built on **ManiSkill 3**. A Franka Panda must pick parcels from
a central inbound zone and place each into the bin whose **color matches the parcel's tag**. Simple
to state, easy to script — hard to make generalize to colors it has never seen.

Routing is purely **relational**: "match the hue," **not** "memorize red → left." Bin sides swap
randomly, so the policy must *read color* to choose where to place. The held-out judge evaluation
uses **reserved hues never seen in training** — same task, same success check, only the color values
and seeds differ.

> **Scoring is method-agnostic.** IL, RL, or hybrid all welcome. The competition is about
> **perception, memory, and color-matching generalization**, not which algorithm you use.

---

## The task

- A Franka Panda arm with a parallel gripper is mounted at a tabletop workstation.
- **Parcels** look like brown warehouse cardboard boxes. Each carries a **colored rectangular tag**
  (a sticker/label) on its **top face**. The **tag color**, not the box color, tells you where the
  parcel goes.
- Parcels spawn in the **inbound zone** in the center of the table, in front of the robot.
- There are two **color-coded output bins**, placed **left and right** of the robot. **Bin sides
  swap randomly between episodes**, so position is not a reliable cue — color is.
- **Goal:** place each parcel into the bin whose color matches the parcel's tag.
- **Score:** fraction of parcels placed in the correct-color bin.

### Color → bin mapping (relational)

A parcel belongs in the bin of the **same hue** as its tag. The mapping is by **hue, not by side**.
Training colors are sampled from a **broad hue distribution** — wide enough that absolute-color
memorization fails and the only winning strategy is to learn the **match-same-hue** relation.

---

## Difficulty levels

Switch with `difficulty=easy|medium|hard`. Difficulty scales by **randomization and parcel count**,
not by adding bins. The episode horizon scales with the parcel count.

| Level  | Parcels | Obs default | Randomization                                                        |
|--------|---------|-------------|----------------------------------------------------------------------|
| easy   | 2       | state*      | fixed poses, none                                                    |
| medium | 3       | rgb         | pose randomization                                                   |
| hard   | 4       | rgb         | RGB + lighting + background + clutter + full appearance randomization|

\* *Easy may be promoted to vision-only — see [Observations](#observations). Medium and hard are
vision-based and are the real challenge.*

**Hard adds, on top of medium:** swapped bin sides, background variation (table/floor), lighting
variation, light clutter, and full appearance randomization (cardboard and tag-color shades). A
**secondary speed metric** (steps to completion) is a tiebreaker only.

---

## The generalization test

Training colors come from a **broad hue distribution**. The **held-out judge evaluation uses
reserved hues never seen in training** to test whether your policy generalizes the matching relation
to new colors. Same task, same success check — only the color values and seeds differ.

**Color never enters scoring** — it lives only in the observation, so judging stays fully
deterministic. We report **seen-hue vs held-out-hue accuracy separately**; the gap between them is
the generalization signal.

A policy that memorizes a fixed layout fails (bins swap sides) and a policy that memorizes absolute
colors fails (held-out hues are unseen). You pass by genuinely **reading the tag, finding the
matching-hue bin, and placing there** — and having that skill hold up on colors you have not seen.

---

## Tracks (suggested, not enforced)

- **Main / IL** — learn from the provided demonstrations (BC → ACT / Diffusion Policy). **This is
  the recommended path.** See [`il/README.md`](il/README.md).
- **Advanced / RL** — no demos; **design your own reward** and policy. The starter ships a
  **sparse reward only**; shaping is yours to build.

---

## Observations

Selectable with `obs_mode=state|rgb`; the default is set per difficulty (above).

### `rgb` (default for the vision levels) — scene camera
- Source: a fixed third-person **scene camera** that keeps the whole workspace (robot, parcels, both
  bins) in frame for the entire episode, so a grasped parcel never occludes the view and the **same
  rgb policy works at any parcel count** (`obs_camera=scene`, the default; `obs_camera=wrist` is also
  available but gets occluded once a parcel is grasped).
- Image tensor at `obs["sensor_data"]["scene_camera"]["rgb"]`, shape `(num_envs, H, W, 3)`,
  **dtype `uint8`, range `[0, 255]`, channel order `RGB`**, on the GPU device. Default `H = W = 128`
  (`camera.width` / `camera.height`).
- If you wrap with ManiSkill's `FlattenRGBDObservationWrapper(rgb=True, depth=False, state=True)` (as
  the starter scripts do) you get `{"rgb": (N,H,W,3) uint8, "state": (N,26) float32}`, where the
  **26-dim `state` is proprioception only** — `agent.qpos` (9) · `agent.qvel` (9) · `tcp_pose`
  (7 = xyz + quat `wxyz`) · `is_grasped` (1). **No** privileged parcel/bin/color info, no depth.

### `state` — privileged low-dim vector (debugging / smallest level)
- A flat `float32` vector: robot proprioception, per-parcel pose, per-parcel **tag hue as raw color
  values** (not a label — so absolute-color memorization still fails), bin positions, and **bin hue
  as raw color values**. The exact layout and dimensionality are documented in the env and scale with
  parcel count.

Judging **locks the obs mode to the difficulty default** — `eval.py` ignores any `obs_mode` override
— so you cannot be judged on privileged `state` at a vision level.

---

## Action space

Fixed `pd_ee_delta_pos` controller, **4 continuous dims, all normalized to `[-1, 1]`**:

| dims | meaning |
|------|---------|
| `[0:3]` | end-effector **delta position** (x, y, z), scaled to ±0.1 m per step (gripper held pointing down) |
| `[3]`   | **gripper**: `+1` = open, `-1` = close |

The action space is fixed — you do not redesign it.

---

## Reward

The starter ships a **sparse** reward only: `+1` when a parcel lands in its correct-color bin, `0`
otherwise. This trains, but slowly.

There is **no example dense reward in the starter**. On the **Advanced / RL** track, designing a
dense/shaped reward is **your** job — it is the core of an RL submission. The IL track does not need
a reward at all (it learns from demonstrations).

You are free to choose your algorithm, network architecture, visual encoder, and (for RL) reward.
You may **not** change the environment, action space, observation interface, or success condition.

---

## Bring your own policy

`eval.py` only requires an object satisfying this contract:

```python
policy.act(obs, deterministic=True) -> Tensor   # shape (num_envs, action_dim), values in [-1, 1]
```

`obs` is **exactly** the environment's observation in the difficulty's locked mode and nothing else —
the policy never sees the env, the ground-truth state, or the scorer, so it can't read privileged
info or game the geometric check. Any RL net, scripted controller, classical CV + control pipeline,
or behavior-cloning model that meets the contract works.

Point `eval.py`/`test.py` at your policy with a loader entrypoint:

```bash
pixi run python eval.py difficulty=hard checkpoint=<path> eval_config=conf/eval/default.yaml \
    policy=my_submission:load_policy
```

```python
# my_submission.py
def load_policy(checkpoint, sample_obs, action_space, device):
    return MyPolicy(checkpoint, sample_obs, action_space, device)   # has .act(obs, deterministic=True)
```

Leave `policy` unset to load the built-in `Agent` from your checkpoint (so a `train.py` checkpoint
runs unchanged). Judging uses the same `policy=` entrypoint.

---

## Scoring

**Primary metric: sort accuracy** — fraction of parcels placed in the correct-color bin, averaged
over the eval episodes. A parcel is correctly sorted when it rests inside the bin whose color matches
its tag. The geometric check, per parcel `j` whose matching bin is centered at `(bx, by)`:

```
|parcel.x - bx| < 0.11 m   AND   |parcel.y - by| < 0.13 m   AND   0 < parcel.z < 0.06 m
```

i.e. the body is inside the bin's 0.22 m × 0.26 m footprint and settled low under the ~0.05 m wall
rim. **Scoring is geometric and deterministic — it does not depend on any camera read or color
value.** A parcel in the *wrong*-color bin is a mis-sort (diagnostic), not a success, and is not
retried.

Reported by the eval/judge harness:
- `sort_accuracy` — **primary ranking metric** (reported **seen-hue vs held-out-hue**)
- `task_progress` — graded partial credit (reach → grasp → carry → place), so near-misses count
- `mean_sorted/episode` — parcels sorted out of total
- `all_placed_rate` — fraction of episodes where every parcel was placed (anywhere)
- `mean_steps` — **speed, tiebreaker only** (lower is better)
- `mis_sort_rate` — wrong-bin placements (diagnostic)

The episode ends when all parcels are placed or the (parcel-count-scaled) step limit is reached.

---

## How judging works — read this

Policies are run by the organizers through a **fixed evaluation harness on held-out configs**, using
**the exact same `eval.py` interface and your trained checkpoint**. Judging:

- runs **all three levels** (easy, medium, hard), each on a **held-out config** that uses **reserved
  hues never seen in training** and a distinct seed list — the task, mapping, and success check are
  identical to what you trained on;
- reports **per-level** `sort_accuracy` and `task_progress`, plus a **weighted average** across
  levels as the headline number;
- reports **seen-hue vs held-out-hue** accuracy side by side — the gap is the generalization signal;
- **locks the obs mode** to each difficulty's default, so privileged `state` can't be used at a
  vision level.

The held-out hues and seeds are **not released**. The visible `conf/eval/default.yaml` is
same-distribution so you can rehearse the exact interface; your real score comes from the held-out
configs. Build for **generalization**, not for the visible distribution: domain randomization,
augmentation, and robust encoders are likely worth more than overfitting the training colors.

Make sure your checkpoint loads and runs under `eval.py` with no code changes — that's exactly what
the judges run.

---

## Quickstart (notebook)

Open **`notebook.ipynb`** (works in Google Colab — select a GPU runtime). It walks the whole
pipeline: install, build the env, view the parcels and bins, inspect the observations, run a short
training, then test and eval, finishing with a rendered rollout. It calls the same scripts and
configs documented below. For the **IL track**, see [`il/notebook_il.ipynb`](il/notebook_il.ipynb)
and [`il/README.md`](il/README.md).

---

## Setup

This repo is [pixi](https://pixi.sh)-managed; all commands run inside the pixi environment.

```bash
pixi install          # create the environment from pixi.toml / pixi.lock
pixi run install      # pip install -e .  (installs the warehouse_sort package + deps)
```

Run any script through pixi, e.g. `pixi run python train.py difficulty=easy`. A CUDA GPU is required
(ManiSkill 3 GPU simulation). Convenience tasks `pixi run train|test|eval` are also defined (append
Hydra overrides, e.g. `pixi run train difficulty=medium`).

---

## Running

### Imitation learning (recommended)
```bash
pixi run python il/gen_demos.py --num-episodes 60 --action-noise 0.05   # generate demos
pixi run python il/train.py dp_rgb                                       # train RGB Diffusion Policy
```
See [`il/README.md`](il/README.md) for the full record → replay → train → eval pipeline and the
per-method policy entrypoints.

### RL (advanced — bring your own reward)
```bash
pixi run python train.py difficulty=medium
pixi run python train.py difficulty=hard num_envs=256 total_steps=5_000_000 seed=1
```
Runs PPO on the **sparse** reward out of the box. Add your own dense reward to go faster — that is
the core of the RL submission. Checkpoints and the resolved config land in `outputs/<date>/<time>/`.

### Test (self-check, same distribution as training)
```bash
pixi run python test.py difficulty=medium checkpoint=outputs/<date>/<time>/ckpt.pt
```

### Eval (same interface as judging)
```bash
pixi run python eval.py difficulty=hard checkpoint=outputs/<date>/<time>/ckpt.pt \
    eval_config=conf/eval/default.yaml
```
`conf/eval/default.yaml` is same-distribution as training, so use it to rehearse the exact interface
judges will use. The **held-out judging configs are not included in this repo** — judges run the same
`eval.py` with your checkpoint against held-out-hue configs across all levels (see *How judging
works*).

---

## Provided

- The **ManiSkill 3 environment**, a **scripted/waypoint reference policy**
  ([`examples/scripted_policy.py`](examples/scripted_policy.py)), and a **demonstration dataset**.
- An **IL starter** (state MLP-BC + RGB Diffusion Policy / ACT) following ManiSkill's standard IL
  pipeline — the recommended path.
- A **sparse reward** for the RL track (design your own dense reward).
- A **Colab-compatible runner** (`notebook.ipynb`, `il/notebook_il.ipynb`).

Held-out judge colors and seeds are **not released**.

---

## Tips (not requirements)

- Get the smallest level running first to confirm your pipeline, then move to the vision levels.
- For vision, a frozen pretrained image encoder is a fast, robust starting point.
- Keep `num_envs` sized to your GPU — RGB observations are the main memory cost.
- Judging is on held-out **hues**: anything that improves color generalization (domain randomization
  over hue, augmentation, robust encoders) is likely worth more than squeezing the training colors.
