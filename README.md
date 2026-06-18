# WarehouseSort — Color-Matching Pick-and-Place Challenge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marso-robotics/berlin-marso-hackathon/blob/main/starter.ipynb)

A robotics imitation-learning challenge built on **[ManiSkill 3](https://maniskill.ai)**.

A Franka Panda robot must sort parcels by color: pick each parcel from the inbound zone and
place it in the bin that matches the **colored tag on its top face** (red tag → red bin, blue
tag → blue bin). At harder levels the bin positions swap between episodes, so the robot must
read the colors rather than memorize a side.

---

## Tracks

| Track | What you submit | Starting point |
|-------|----------------|----------------|
| **Main — State IL** | A Diffusion Policy trained on the provided demonstrations, using the privileged state observation | **Works out of the box (~85% on easy)** |
| Advanced — Image IL | A Diffusion Policy trained on the provided demonstrations, using only camera images | Template provided — **not yet solving the task** |
| Bonus — RL | Any policy trained from scratch using the sparse `+1` reward | No starter provided; design your own reward |

**Start with the Main track.** The state IL baseline already closes the loop on easy. The
challenge is to generalize across difficulty levels and to unseen layouts (new positions,
bin side assignments) that appear in the held-out judge configs.

---

## Difficulty levels

| Level | Parcels | Randomization |
|-------|---------|--------------|
| **easy** | 2 | Fixed parcel positions, fixed bin sides |
| **medium** | 4 | Randomized parcel positions within the inbound zone |
| **hard** | 6 | Randomized positions + bin sides may swap between episodes |

The held-out judge configs use the same difficulty levels but with different seeds and
slightly wider position randomization — build for generalization, not for the exact training layout.

---

## Scoring

### Primary metric — Sort accuracy

**Sort accuracy** is the fraction of parcels placed in the correct-color bin by episode end.
It is the **only metric that determines your ranking**.

The check is **geometric and deterministic**: a parcel is correctly sorted when its body
rests inside the footprint of the matching-color bin and is settled low (below the rim).

### Final score — weighted average across all three levels

| level | weight |
|-------|--------|
| easy | 0.2 |
| medium | 0.3 |
| hard | 0.5 |

```
final_score = 0.2 × sort_accuracy_easy
            + 0.3 × sort_accuracy_medium
            + 0.5 × sort_accuracy_hard
```

Higher weights on harder levels reward generalisation. The tiebreaker (hard level only) is
`mean_steps` — fewer steps for the same accuracy is better.

---

## Quick start

```bash
# 1. Install
pixi install
pixi run install        # pip install -e .

# 2. Generate demonstrations (easy level, 60 episodes)
pixi run python il/gen_demos.py --num-episodes 60 --action-noise 0.05

# 3. Train state Diffusion Policy
pixi run python il/train.py method=dp

# 4. Evaluate
pixi run python eval.py difficulty=easy obs_mode=state \
    policy=warehouse_sort.il_policy:load_dp \
    checkpoint=il/baselines/diffusion_policy/runs/warehouse_state_dp_easy/checkpoints/best_eval_success_at_end.pt \
    eval_config=conf/eval/default.yaml
```

For a guided walkthrough, open **[starter.ipynb](starter.ipynb)** — or click the badge at the top of this page to launch it directly in Google Colab (select a GPU runtime).

---

## Observations

### State (main track)
A flat `float32` vector, shape `(num_envs, 54)` for easy (2 parcels):

| slice | field | dims |
|-------|-------|------|
| `[0:9]` | joint positions (`qpos`) | 9 |
| `[9:18]` | joint velocities (`qvel`) | 9 |
| `[18:25]` | TCP pose (xyz + quat wxyz) | 7 |
| `[25:26]` | is_grasped | 1 |
| `[26:40]` | parcel poses (xyz + quat per parcel) | P×7 |
| `[40:44]` | parcel tag colors (one-hot [red, blue]) | P×2 |
| `[44:50]` | bin positions (xyz of red bin, blue bin) | 2×3 |
| `[50:54]` | bin color one-hot | 2×2 |

Total dim = `26 + P×7 + P×2 + 6 + 4` (= 54 for P=2).

### RGB (advanced track — image template, not yet working)
A fixed third-person scene camera. With `FlattenRGBDObservationWrapper`:
- `obs["rgb"]`: `(N, 128, 128, 3)` uint8, RGB channel order
- `obs["state"]`: `(N, 26)` float32 — proprioception only (no parcel/bin info)

---

## Action space

`pd_ee_delta_pos`, 4 dims in `[-1, 1]`:

| dims | meaning |
|------|---------|
| `[0:3]` | end-effector delta xyz (±0.1 m/step) |
| `[3]` | gripper: +1 = open, −1 = close |

---

## Reward

**Sparse only: `+1` per correctly placed parcel.** No dense reward is provided. On the RL
bonus track, designing a shaped reward is your job.

---

## How judging works

The judges run `judge/run_judge.py`, which evaluates your checkpoint on all three levels and
reports a weighted aggregate:

```bash
python judge/run_judge.py \
    checkpoint=<your_ckpt> \
    policy=warehouse_sort.il_policy:load_dp
```

Internally it calls the same `eval.py` interface per level with held-out configs
(`judge/heldout_easy.yaml`, `judge/heldout_medium.yaml`, `judge/heldout_hard.yaml`).

The held-out configs use the same difficulty levels, same colors, and same success check —
only the seeds and position randomization ranges differ (slightly wider than training).
The observation mode is locked to the difficulty default (state for all three main-track
levels), so you cannot pass privileged info at eval. Your checkpoint must load and run with
no code changes.

---

## References

- **ManiSkill 3** — GPU-accelerated robot simulation: [maniskill.ai](https://maniskill.ai) / [arxiv.org/abs/2410.00425](https://arxiv.org/abs/2410.00425)
- **Diffusion Policy** — Chi et al. 2023: [diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu)
- The RGB template is built on the ManiSkill IL baselines and **[LeRobot](https://github.com/huggingface/lerobot)** conventions.

---

## Repo layout

```
warehouse_sort/     # the ManiSkill environment + IL policy entrypoints
  env.py            # WarehouseSort-v1 (register, scene, obs, reward, evaluate)
  il_policy.py      # load_dp / load_dp_rgb — wire into eval.py via policy=...
  utils.py          # env construction, rollout, metrics printing

conf/               # Hydra configs
  difficulty/       # easy.yaml / medium.yaml / hard.yaml
  eval/default.yaml # same-distribution eval (rehearse the judge interface)

il/                 # imitation learning
  gen_demos.py      # record + replay demonstrations
  train.py          # Hydra dispatcher -> vendored Diffusion Policy trainer
  baselines/        # vendored ManiSkill DP baseline (diffusion_policy only)

examples/
  scripted_policy.py  # deterministic waypoint policy (demo source)

eval.py             # evaluate a checkpoint (same interface as judging)
test.py             # self-check on same-distribution episodes
judge/              # held-out eval configs (not distributed to competitors)
```
