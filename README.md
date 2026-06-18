# WarehouseSort — Color-Matching Pick-and-Place

An **imitation-learning challenge** built on **ManiSkill 3**. A Franka Panda picks parcels from a
central inbound zone and places each into the bin whose **color matches the parcel's tag** (red tag →
red bin, blue tag → blue bin). Bin sides swap between episodes, so the policy must read the tag color,
not memorize a side.

The challenge is **generalization to new layouts**: difficulty scales with **how much the parcel
positions are randomized** and **how many parcels there are to sort**. The held-out judge evaluation
uses **unseen positions, bin arrangements, and seeds** — same task, same colors, same success check,
only the layout and seeds differ.

> **This is an IL challenge.** The main track is **behavior cloning from the provided demonstrations
> on the privileged state vector** — the tractable path that already closes the loop. Learning from
> images, and learning from reward (RL), are the **advanced / open** track: known to be hard, and not
> yet solved on this task.

---

## The task

- A Franka Panda arm with a parallel gripper is mounted at a tabletop workstation.
- **Parcels** look like brown warehouse cardboard boxes. Each carries a **colored rectangular tag**
  (red or blue) on its **top face**. The **tag color**, not the box color, tells you where it goes.
- Parcels spawn in the **inbound zone** in the center of the table, in front of the robot.
- There are two **color-coded output bins** (a red bin and a blue bin), placed **left and right** of
  the robot. **Bin sides swap between episodes**, so position is not a reliable cue — color is.
- **Goal:** place each parcel into the bin whose color matches the parcel's tag.
- **Score:** fraction of parcels placed in the correct-color bin.

### Color → bin mapping

Red tag → red bin, blue tag → blue bin. The mapping is by **color, not by side**: because the bins'
left/right positions swap between episodes, you must identify the destination by color, not by a
memorized location. The color palette is **fixed** (red/blue) — color is the routing cue, not a
generalization axis.

---

## Tracks

### Main track — Imitation Learning on **state** (recommended, this is the challenge)

Learn a policy by **behavior cloning the provided demonstrations** using the privileged low-dim
**state** observation (parcel poses, tag colors, bin positions and colors, proprioception). This is
the path that works: a Diffusion Policy on state already closes the loop on the seen distribution.
**Your job is to make it generalize to unseen layouts** (new positions, bin arrangements) across the
three difficulty levels.

```bash
pixi run python il/gen_demos.py --num-episodes 60 --action-noise 0.05   # generate demos
pixi run python il/train.py method=dp                                    # Diffusion Policy on state
```

See [`il/README.md`](il/README.md) for the full record → replay → train → eval pipeline.

### Advanced / open track — known to be hard, not yet solved

For teams who want the research frontier. **These are currently known *not* to work well on this
task** — that's the point. Bonus territory, not the main scoreboard:

- **Imitation learning from images** — RGB instead of state. The wrist camera is occluded by a
  grasped parcel; a fixed scene camera helps but vision is still open.
- **Reinforcement learning on state** — no demos. The starter ships a **sparse reward only**;
  designing a dense/shaped reward that actually converges is your job.

---

## Difficulty levels

Switch with `difficulty=easy|medium|hard`. Difficulty scales with **position randomization** and the
**number of parcels to sort** — nothing else. The episode horizon scales with parcel count.

| Level  | Parcels | Position randomization                                |
|--------|---------|------------------------------------------------------|
| easy   | 2       | fixed parcel poses, fixed bin sides                  |
| medium | 3       | randomized parcel poses; bin sides may swap          |
| hard   | 4       | randomized parcel poses + light clutter; bins swap   |

At **every** level the **held-out judge config widens the position randomization** (larger spawn
jitter, full bin-side swaps) and uses **unseen seeds** — that is the generalization test. A
**secondary speed metric** (steps to completion) is a tiebreaker only.

---

## The generalization test

Training layouts come from a configured randomization range (spawn jitter, bin-side swaps). The
**held-out judge evaluation widens that randomization and uses reserved seeds never seen in
training** — testing whether your policy handles parcel positions and bin arrangements it has not
seen. Same task, same colors, same success check — only the layout and seeds differ.

A policy that memorizes a fixed layout fails (positions jitter and bins swap sides). You pass by
genuinely learning to **read the tag, find the matching-color bin wherever it is, and place there** —
and having that hold up on layouts you have not seen. We report **seen-layout vs held-out-layout**
accuracy; the gap is the generalization signal.

---

## Observations

Selectable with `obs_mode=state|rgb`. The **main track uses `state`**; `rgb` is the advanced track.

### `state` (main track) — privileged low-dim vector
- A flat `float32` vector. Layout (P = parcel count), e.g. dim 54 for easy (P=2):

  | slice | field | dims | meaning |
  |-------|-------|------|---------|
  | proprio | `qpos`·`qvel`·`tcp_pose`·`is_grasped` | 26 | 9 + 9 + 7 + 1 |
  | `parcel_pose` | per parcel xyz + quat `wxyz` | `P×7` | parcel `j` at index `j` |
  | `parcel_tag` | per-parcel tag color one-hot `[red, blue]` | `P×2` | which bin color to match |
  | `bin_position` | xyz of red bin then blue bin | `2×3` | current left/right positions |
  | `bin_color` | bin color one-hot (identity) | `2×2` | color id of each bin |

  Total dim `= 26 + P*7 + P*2 + 6 + 4`. Parcel `j`'s correct destination is the bin whose color id
  matches `parcel_tag[j]`. Ordering is documented in the env.

### `rgb` (advanced track) — scene camera
- A fixed third-person **scene camera** (`obs_camera=scene`) keeps the whole workspace in frame for
  the whole episode (a grasped parcel never occludes it) and is parcel-count-agnostic. Image tensor
  `(num_envs, H, W, 3)`, **uint8, `[0,255]`, channel order `RGB`**, default `H=W=128`. With
  `FlattenRGBDObservationWrapper(rgb=True, depth=False, state=True)` you get
  `{"rgb": (N,H,W,3) uint8, "state": (N,26) f32}` where the 26-d state is **proprioception only** (no
  privileged parcel/bin/color, no depth).

Judging **locks the obs mode to the difficulty default** — `eval.py` ignores any `obs_mode`
override — so a policy is judged in the mode its track defines.

---

## Action space

Fixed `pd_ee_delta_pos` controller, **4 continuous dims, all normalized to `[-1, 1]`**:

| dims | meaning |
|------|---------|
| `[0:3]` | end-effector **delta position** (x, y, z), scaled to ±0.1 m per step (gripper held pointing down) |
| `[3]`   | **gripper**: `+1` = open, `-1` = close |

The action space is fixed — you do not redesign it.

---

## Demonstrations

The main track learns from a **provided demonstration dataset** generated by the scripted waypoint
policy ([`examples/scripted_policy.py`](examples/scripted_policy.py)). The demos:

- cover **all levels** (easy / medium / hard) plus a **mixed** set, recorded in ManiSkill's standard
  `.h5` + `.json` trajectory format (record → `replay_trajectory` → train);
- use **randomized pick order** per episode (the dataset does not always sort in the same sequence);
- include **realistic imperfections** — failed picks and retries — but **never a mis-sort** (a parcel
  is only ever released over its correct-color bin), so destination labels stay clean.

Regenerate or extend them with `il/gen_demos.py`; see [`il/README.md`](il/README.md).

---

## Reward (advanced / RL only)

The starter ships a **sparse** reward only: `+1` when a parcel lands in its correct-color bin, `0`
otherwise. The **main IL track does not need a reward** (it learns from demonstrations). On the
**advanced RL track**, designing a dense/shaped reward that converges is **your** job — there is no
example dense reward in the starter.

You may **not** change the environment, action space, observation interface, or success condition.

---

## Bring your own policy

`eval.py` only requires an object satisfying this contract:

```python
policy.act(obs, deterministic=True) -> Tensor   # shape (num_envs, action_dim), values in [-1, 1]
```

`obs` is **exactly** the environment's observation in the difficulty's locked mode and nothing else —
the policy never sees the env, the ground-truth state, or the scorer, so it can't read privileged
info or game the geometric check. Any BC/DP/ACT model, RL net, scripted controller, or classical
pipeline that meets the contract works.

Point `eval.py`/`test.py` at your policy with a loader entrypoint:

```bash
pixi run python eval.py difficulty=easy obs_mode=state checkpoint=<path> \
    eval_config=conf/eval/default.yaml policy=warehouse_sort.il_policy:load_dp
```

```python
# my_submission.py
def load_policy(checkpoint, sample_obs, action_space, device):
    return MyPolicy(checkpoint, sample_obs, action_space, device)   # has .act(obs, deterministic=True)
```

Leave `policy` unset to load the built-in `Agent` from your checkpoint. Judging uses the same
`policy=` entrypoint.

---

## Scoring

**Primary metric: sort accuracy** — fraction of parcels placed in the correct-color bin, averaged
over the eval episodes. A parcel is correctly sorted when it rests inside the bin whose color matches
its tag. The geometric check, per parcel `j` whose matching bin is centered at `(bx, by)`:

```
|parcel.x - bx| < 0.11 m   AND   |parcel.y - by| < 0.13 m   AND   0 < parcel.z < 0.06 m
```

i.e. the body is inside the bin's 0.22 m × 0.26 m footprint, settled low under the ~0.05 m wall rim.
**Scoring is geometric and deterministic.** A parcel in the *wrong*-color bin is a mis-sort
(diagnostic), not a success, and is not retried.

Reported by the eval/judge harness:
- `sort_accuracy` — **primary ranking metric** (reported **seen-layout vs held-out-layout**)
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

- runs **all three levels** (easy, medium, hard), each on a **held-out config** that widens the
  position randomization (larger spawn jitter, full bin-side swaps) and uses a **distinct seed
  list** — the task, colors, mapping, and success check are identical to what you trained on;
- reports **per-level** `sort_accuracy` and `task_progress`, plus a **weighted average** across
  levels as the headline number;
- reports **seen-layout vs held-out-layout** accuracy side by side — the gap is the generalization
  signal;
- **locks the obs mode** to each difficulty's default (state for the main track).

The held-out seeds and ranges are **not released**. The visible `conf/eval/default.yaml` is
same-distribution so you can rehearse the exact interface; your real score comes from the held-out
configs. Build for **generalization across layouts**, not for the visible distribution.

Make sure your checkpoint loads and runs under `eval.py` with no code changes — that's exactly what
the judges run.

---

## Quickstart (notebook)

For the IL track, open [`il/notebook_il.ipynb`](il/notebook_il.ipynb) (works in Google Colab — select
a GPU runtime); see [`il/README.md`](il/README.md). The root `notebook.ipynb` walks the env, the
observations, a short training, then test and eval, finishing with a rendered rollout.

---

## Setup

This repo is [pixi](https://pixi.sh)-managed; all commands run inside the pixi environment.

```bash
pixi install          # create the environment from pixi.toml / pixi.lock
pixi run install      # pip install -e .  (installs the warehouse_sort package + deps)
```

A CUDA GPU is required (ManiSkill 3 GPU simulation). Convenience tasks `pixi run train|test|eval` are
defined (append Hydra overrides).

---

## Running

### Main track — state IL
```bash
pixi run python il/gen_demos.py --num-episodes 60 --action-noise 0.05   # demos
pixi run python il/train.py method=dp                                    # Diffusion Policy on state
pixi run python eval.py difficulty=easy obs_mode=state checkpoint=<ckpt> \
    eval_config=conf/eval/default.yaml policy=warehouse_sort.il_policy:load_dp
```

### Advanced — image IL / state RL (open problems)
```bash
pixi run python il/train.py method=dp_rgb        # image (scene-cam) Diffusion Policy — open
pixi run python train.py difficulty=medium       # PPO on sparse reward — bring your own reward
```

### Test / Eval
```bash
pixi run python test.py difficulty=medium checkpoint=outputs/<date>/<time>/ckpt.pt
pixi run python eval.py difficulty=hard checkpoint=<ckpt> eval_config=conf/eval/default.yaml
```
`conf/eval/default.yaml` is same-distribution as training (rehearse the interface). The **held-out
judging configs are not included** — judges run the same `eval.py` against held-out layout configs
across all levels.

---

## Provided

- The **ManiSkill 3 environment**, a **scripted/waypoint reference policy**, and a **demonstration
  dataset** (easy / medium / hard + mixed; randomized order; realistic imperfections, never a
  mis-sort).
- An **IL starter** (state MLP-BC + state Diffusion Policy / ACT; RGB DP as the advanced template)
  following ManiSkill's standard IL pipeline.
- A **sparse reward** for the advanced RL track (design your own dense reward).
- A **Colab-compatible runner**.

Held-out judge seeds and ranges are **not released**.

---

## Tips (not requirements)

- Get `difficulty=easy` (state) running first to confirm your pipeline, then move up.
- The main-track challenge is **generalizing to unseen layouts**, not perception — anything that
  helps the policy handle new positions and bin arrangements (position augmentation, randomized
  training layouts) is likely worth the most.
- Keep `num_envs` sized to your GPU.
- Image IL and state RL are the **open** track — expect them to be hard.
