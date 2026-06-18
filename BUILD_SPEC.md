# BUILD SPEC — Warehouse Colour-Sort Pick-and-Place (Hackathon Starter)

This is a build brief for a Claude Code agent. Scaffold a hackathon starter repo from this
specification. Build the **environment, observation interface, action space, success conditions, and
runnable scripts** — but **do not write a dense reward**. The default reward is sparse (see below);
reward shaping is the participants' job.

---

## 0. Overriding build principles

- **Simple and readable over clever.** One env file, one config tree, a small number of scripts. No
  registries, no plugin systems, no multi-simulator abstraction layers.
- **Specify the problem, not the solution.** The deliverable defines the environment, observations,
  actions, and success condition. It must NOT contain reward shaping, solution hints, or a worked
  policy.
- **Difficulty is a pure config switch.** Changing difficulty must never require editing script or
  env logic — only config values.
- **State first to verify plumbing, then vision is the real task.** See difficulty levels.

---

## 1. Task

A Franka Panda arm picks parcels from an inbound zone and places each into the output bin that
matches the parcel's colour (e.g. red parcels → left bin, blue → right bin). The episode score is
the number of parcels correctly sorted.

This is a scoped each-picking / order-sortation task: the full perceive → decide → grasp → place
loop, kept demoable within a hackathon.

---

## 2. Simulator

Use **ManiSkill 3** (Gymnasium API, GPU-parallel, built-in Franka Panda and pick-and-place
references). Do not abstract across simulators.

- Copy the attachment/grasp mechanics from the built-in **`PickCube`** environment rather than
  inventing a new grasp mechanism. Use `PickCube` as the structural reference for the env too
  (`_load_scene`, `_initialize_episode`, `evaluate`, reward signature) and mirror its grasp-detection
  and object-attachment pattern.
- Prefer GPU paths (`num_envs > 1`, `device="cuda"`).

---

## 3. Environment specification

One registered env subclass of `BaseEnv`, parameterised by difficulty. Implements `_load_scene`,
`_initialize_episode`, `evaluate`, and a **sparse** `compute_dense_reward` (see §6).

### Scene

- **Robot:** Franka Panda with default parallel gripper, default controller, mounted at a tabletop.
- **Parcels:** small box primitives that look like **warehouse brown cardboard**. Colour is NOT the
  parcel body — it is a **rectangular tag/sticker on the top face** of each parcel (e.g. red or blue),
  like a shipping label. The policy must detect the tag, not the box colour.
- **Bins:** one output bin per tag colour, **colour-coded** (a red bin, a blue bin, etc.) so the
  correct destination is identifiable from the camera. Bins are **low-walled** (minimal height, to
  avoid excessive collision) and **wide enough to hold all parcels**. Placed to the **left and right**
  of the robot.
- **Layout:** tabletop with the **parcels in the centre** in front of the robot, **bins on the left
  and right**.
- **Inbound zone:** a defined rectangular region in the centre of the tabletop where parcels start.

### Colour → bin mapping

The tag colour determines the destination bin (e.g. red tag → red bin). The mapping is by **bin
colour, not bin position** — see the bin-swap randomisation at hard difficulty (§3, hard). Bins are
colour-coded so the mapping is always visually determinable.

### Difficulty levels (single config switch `difficulty: easy | medium | hard`)

| Level  | Parcels | Poses              | Tag colours / Bins | Obs default | Randomisation        |
|--------|---------|--------------------|--------------------|-------------|----------------------|
| easy   | 2       | fixed              | 2 / 2              | **state**   | none                 |
| medium | 4–6     | randomised, no overlap, **tag always top-visible** | 2 / 2 | **rgb** | slight pose randomisation only |
| hard   | 6–8     | randomised, light clutter (touching ok, no stacking/occlusion), **tag always top-visible** | 2 / 2 | **rgb** | see hard list below |

**Easy** — plumbing check only, not a demo level. Confirms the loop closes and the interface works.
State obs (tag colour handed over directly as a label).

**Medium** — the first real (vision) level. Parcels in randomised positions/orientations within the
inbound zone, but **every parcel's tag must remain visible from the top-down/wrist camera** (no
orientation that hides or occludes the tag). Two tag colours, two colour-coded bins. No appearance
randomisation.

**Hard** — vision under stress, and the **generalisation level**. Every randomisation axis is
present in training but at **narrow ranges**; the held-out judging config widens and recombines those
same axes (see §8 generalisation principle and §10). On top of medium's randomisation, hard adds, all
**present-but-narrow in training**:
- **Bin position:** the left/right placement of the colour-coded bins is randomised **during
  training** (narrow — e.g. small left/right jitter, and the red/blue sides may swap a fraction of
  episodes). Because bins move in training, a policy cannot memorise "red → left"; it must read bin
  colour. Held-out widens this (full swaps, larger position offsets).
- **Background:** table and floor colours varied over a **small training palette**. Held-out uses
  unseen colours.
- **Lighting:** intensity/direction varied over a **narrow training range**. Held-out widens it.
- **Appearance:** cardboard shades and tag-colour shades varied **slightly** in training. Held-out
  uses unseen shades (still within the same colour identity — a red tag is always recognisably red).
- 2 tag colours / 2 colour-coded bins, light clutter (same as medium; hard differs by randomisation, not bin count).

The principle (see §8): **every axis the policy must be robust to is exercised in training at a
narrow range, so the capability is learnable; held-out tests whether that capability generalises to
unseen ranges and combinations.** No axis is held-out-only — that would test a cue the policy was
never given a reason to learn.

Keep clutter light at all levels. No chaotic piles, no occlusion, and the tag is always top-visible.
Full bin-picking is out of scope.

---

## 4. Observation interface (both modes defined; default per level)

Expose **both** observation modes, selectable by config (`obs_mode: state | rgb`). Document exact
shapes and dtypes in the README so they are unambiguous.

### `rgb` (default for medium/hard)
- Source: **Panda wrist camera** (the default ManiSkill Panda wrist-mounted RGB camera).
- Resolution: config field (`camera.width`, `camera.height`), default small (e.g. 128×128) so it
  runs on modest GPUs.
- Returned as a torch tensor on device, shape `(num_envs, H, W, 3)` (confirm channel order against
  ManiSkill convention and document it).
- May include proprioception (joint pos/vel, gripper state, TCP pose) alongside the image, as
  ManiSkill's `rgb` obs mode normally does — document exactly what is included.

### `state` (default for easy)
- Privileged low-dimensional vector: parcel poses, parcel **tag colours** (as labels/one-hot), bin
  positions **and bin colours**, robot proprioception. Document the exact layout and ordering.

Participants choose how to consume these. Do not prescribe an encoder, architecture, or algorithm.

---

## 5. Action space (defined and fixed)

- Franka arm controller (e.g. `pd_ee_delta_pose` or `pd_joint_delta_pos` — pick one, document it) +
  gripper open/close dimension.
- Fixed. Participants do not redesign the action space.
- Document the exact dimensionality and the meaning of each dimension in the README.

---

## 6. Reward (sparse default + a simple worked example)

- **Default reward is sparse:** `+1` each time a parcel is placed into its correct bin, `0`
  otherwise. Derived directly from the success condition in `evaluate`. This is what runs unless a
  participant replaces it.
- **Also provide one example dense reward** as a reference, clearly separated and marked optional:
  a simple, normalised dense reward in the style of ManiSkill's `PickCube` (reach distance to parcel
  → grasp → distance to correct bin → placement), operating from **state**. It should be the
  **simplest readable version that solves the task from state — not optimised for speed or sample
  efficiency.** Its purpose is to show participants a working shaping pattern they can learn from and
  improve on, not to be a strong baseline.
- Ship the example as a clearly-labelled alternative (e.g. a separate `reward.py` function or a config
  flag `reward=sparse|example_dense`), **defaulting to sparse**. Mark it:
  `# EXAMPLE DENSE REWARD (state-based, ManiSkill PickCube style). Simplest readable version, not
  optimised. Replace/improve with your own — this is the core of your submission.`
- Do NOT provide a vision-based or tuned dense reward. The competition is designing a better reward
  (and policy), especially for the vision levels.

---

## 7. Success and termination (defined and visible)

In `evaluate`, compute per-env:

- **Correct placement:** a parcel is correctly sorted when its body rests inside the bin whose colour
  matches the parcel's **tag colour** (within the bin footprint, below a height threshold, roughly
  settled). The check is **geometric** (parcel position inside the correct-colour bin region), not a
  camera read — scoring must be deterministic and ungameable. The scene is constructed so this is
  *also* visually verifiable from the top camera (coloured tags visibly inside the matching coloured
  bin), but the score itself does not depend on a vision read.
- **Primary score:** integer count of correctly-sorted parcels in the episode. This is the headline
  metric at all levels.
- **Secondary metric (hard only): speed.** Report steps-to-complete (or parcels-sorted-per-step) as a
  secondary metric at hard difficulty. Primary ranking is still correct-sort count; speed is a
  tiebreaker / secondary leaderboard, so the headline number stays clean.
- **Termination:** episode ends when all parcels are placed OR a step limit is reached
  (`max_episode_steps`, config field). A misplaced parcel (wrong-colour bin) does not count toward
  score and may be defined as not retryable for simplicity — document the choice.
- `evaluate` returns at least: `{"success_count": <int per env>, "all_placed": <bool per env>}`, plus
  `{"steps_to_complete": <int per env>}` at hard.

The success condition is fully visible to participants — they need it to know what they optimise
toward. It is the definition of success, not a reward.

---

## 8. Config (Hydra, plain YAML) + the generalisation principle

### Generalisation principle (governs all randomisation)

**Every axis the policy must be robust to is present in TRAINING at a narrow range, and the HELD-OUT
judging config widens and/or recombines those same axes. No axis is held-out-only.** Train on the
axes; judge on unseen ranges and combinations along them. This makes hard a true generalisation test
(the capability is learnable in training, and judging measures whether it transfers) rather than a
surprise probe for a cue the policy was never given a reason to learn.

Every randomisation axis must therefore be expressed as an explicit, overridable **range** in config
— never hardcoded — so the same schema describes both the narrow training distribution and the wide
held-out distribution by changing values only.

### Config tree

```
conf/                          # COMPETITOR-FACING (shipped in the repo they get)
  config.yaml                  # defaults list + shared params (seed, num_envs, total_steps, device, reward)
  difficulty/
    easy.yaml                  # parcels, fixed poses, obs_mode: state, all rand ranges empty/zero
    medium.yaml                # obs_mode: rgb (wrist cam), pose rand range (narrow), no appearance rand
    hard.yaml                  # obs_mode: rgb, 2 colours, clutter, ALL rand axes present at narrow ranges
  randomization/
    train.yaml                 # the narrow training ranges for every axis (referenced by hard.yaml)
  eval/
    default.yaml               # VISIBLE: same distribution as training (narrow), different seeds

judge/                         # JUDGE-ONLY — NOT shipped to competitors (see §10)
  heldout.yaml                 # the actual held-out eval config: widened/recombined ranges, real values
```

The `judge/` directory is **excluded from the competitor-facing repo** (gitignored, and not included
in whatever archive/branch competitors receive). It is delivered separately to judges. It uses the
exact same schema and the exact same `eval.py`; only the file location and the values differ.

### Randomisation schema (used by both training and eval configs)

Every axis is a named, overridable field with explicit ranges. The agent defines this schema once and
both `randomization/train.yaml` and the eval configs populate it. Required axes:

```yaml
randomization:
  parcel_pose:
    xy_jitter: [<m>, <m>]          # +/- range in metres within inbound zone
    yaw_jitter: [<rad>, <rad>]     # tag must remain top-visible
  bin_position:
    side_swap_prob: <float>        # P(red/blue sides swapped) — narrow in train, wide in heldout
    xy_jitter: [<m>, <m>]          # per-bin position jitter
  lighting:
    intensity: [<lo>, <hi>]
    direction_jitter: [<rad>, <rad>]
  background:
    table_colors: [<list of rgb>]  # small palette in train, unseen colors in heldout
    floor_colors: [<list of rgb>]
  appearance:
    cardboard_shade: [<lo>, <hi>]  # shade variation, same brown identity
    tag_shade: [<lo>, <hi>]        # shade variation, same color identity (red stays red)
  clutter:
    enabled: <bool>
    contact_ok: <bool>             # parcels may touch, never stack/occlude
```

- `difficulty` is the top-level switch. `obs_mode`, `camera.*`, `num_envs`, `reward` are config fields.
- `reward=sparse|example_dense` selects the reward (default `sparse`).
- easy sets all ranges to zero/empty (deterministic). medium populates `parcel_pose` only. hard
  populates every axis at narrow training ranges (from `randomization/train.yaml`).
- Override anything from the CLI. Log resolved config + git hash at the start of every run.

---

## 9. Scripts (the team-facing interface)

Three scripts, consistent CLI, all run through the project's environment manager.

### `train.py`
Runs RL training at the chosen difficulty on the **train** distribution (narrow randomisation ranges,
visible seeds). Saves a checkpoint to the run output dir.

```bash
python train.py difficulty=medium
python train.py difficulty=hard num_envs=256 total_steps=5_000_000
```

### `test.py`
Loads a checkpoint and runs it on **same-distribution** held-back episodes (different seeds, same
ranges) so teams self-check progress. Prints the metrics (§9.1).

```bash
python test.py difficulty=hard checkpoint=outputs/<date>/<time>/ckpt.pt
```

### `eval.py`  ← **interface must exactly match the held-out judging harness**
Loads a checkpoint and runs it on an eval config, reporting the metrics in §9.1 aggregated over N
episodes.

```bash
python eval.py difficulty=hard checkpoint=<path> eval_config=conf/eval/default.yaml
```

**Critical requirements:**
- `eval.py` is **fully driven by the `eval_config` file path** passed on the CLI. Everything that
  defines the eval instances — all randomisation ranges (the §8 schema), seeds, and `n_episodes` —
  is read from that file. Nothing about the eval conditions is hardcoded in the script.
- A checkpoint trained by `train.py` must load and run under `eval.py` with **no code changes**.
- The observation mode used at eval is **locked to the difficulty default** (state for easy, rgb for
  medium/hard) regardless of what the team trained with — so a team cannot train medium/hard on
  privileged `state` and be judged on it. Enforce this in `eval.py`.
- Ships with a **visible** `conf/eval/default.yaml` (narrow, same as training) so teams run `eval.py`
  themselves. The **actual held-out config lives in `judge/heldout.yaml`** (concrete widened values),
  is **not shipped to competitors** (§10), and is run by judges via `eval_config=judge/heldout.yaml`.
  Script and checkpoint interface are identical for both; only the config file (and its values) differ.

### 9.1 Metrics (computed by `evaluate`, aggregated by `test.py` / `eval.py`)

`evaluate` returns per-env per-step the raw quantities; the scripts aggregate over `n_episodes` and
print a summary. Required:

- **`success_count`** (primary) — number of parcels in the correct-colour bin at episode end.
  Reported as **mean correctly-sorted parcels per episode** and **sort accuracy** = sorted / total
  parcels.
- **`all_placed`** (bool) — whether every parcel was placed (correctly or not).
- **`steps_to_complete`** (hard secondary / speed) — steps until all parcels placed, or `max_episode_steps`
  if not all placed. Reported as mean over episodes. Speed is a **tiebreaker only**; `success_count`
  is always the primary ranking metric.
- **`mis_sort_count`** — parcels placed in a wrong-colour bin (diagnostic; not scored).

`eval.py` prints a single clear summary block, e.g.:
```
EVAL  difficulty=hard  n_episodes=200  obs_mode=rgb
  sort_accuracy:        0.83        # PRIMARY
  mean_sorted/episode:  4.98 / 6
  all_placed_rate:      0.91
  mean_steps:           412         # speed (hard tiebreaker)
  mis_sort_rate:        0.06        # diagnostic
```
Aggregation must be deterministic given the eval config's seed list, so re-running the same config +
checkpoint reproduces the numbers.

---

## 10. Held-out eval — defined now, kept out of competitor view, run by judges

The agent **authors a concrete, fully-defined held-out eval config** at `judge/heldout.yaml` with
real widened values (not placeholders). It is run by judges through the **same `eval.py`** with the
team's checkpoint:

```bash
python eval.py difficulty=hard checkpoint=<team_ckpt> eval_config=judge/heldout.yaml
```

### Separation requirements (critical)

- `judge/heldout.yaml` (and the whole `judge/` dir) must be **excluded from what competitors receive**:
  add `judge/` to `.gitignore`, and if competitors get an archive or a branch, exclude `judge/` from
  it. Competitors see the schema (via `conf/eval/default.yaml`) and the README's description of the
  generalisation axes, but never the held-out values.
- The held-out config must run with **zero code changes** — identical pipeline, identical checkpoint
  interface, obs-mode locked to the difficulty default (§9). The only difference from
  `conf/eval/default.yaml` is the randomisation values and the seed list.
- Keep the **task identical**: same robot, same tag-colour → bin-colour mapping, same success check,
  same metrics (§9.1). Only the §8 randomisation ranges widen/recombine.

### Concrete held-out definition

The held-out config widens every training axis and recombines them. The agent sets exact numbers
**coherent with the actual train ranges it chose** (held-out must be strictly wider / unseen relative
to `randomization/train.yaml`). Target relationship per axis:

```yaml
# judge/heldout.yaml  (hard difficulty; JUDGE-ONLY)
randomization:
  parcel_pose:
    xy_jitter:   [~2-3x the train range]      # wider spawn area within the inbound zone
    yaw_jitter:  [~2-3x the train range]      # still constrained so the tag stays top-visible
  bin_position:
    side_swap_prob: 0.5                        # train ~0.1 -> heldout 0.5 (full uncertainty)
    xy_jitter:   [~2-3x the train range]       # larger per-bin position offsets
  lighting:
    intensity:   [wider than train, incl. dimmer + brighter than seen]
    direction_jitter: [~2x train]
  background:
    table_colors: [colours NOT in the train palette]
    floor_colors: [colours NOT in the train palette]
  appearance:
    cardboard_shade: [shades outside train range, still recognisably brown]
    tag_shade:       [shades outside train range, still recognisably the same colour identity]
  clutter:
    enabled: true
    contact_ok: true
eval:
  n_episodes: <e.g. 200>
  seeds: [<fixed list distinct from train/test seeds>]
```

- The held-out seed list must be **distinct** from the train and `conf/eval/default.yaml` seeds.
- `n_episodes` large enough for a stable mean (e.g. 200).
- Values must keep colour/material **identity** intact (a red tag is always recognisably red, cardboard
  always recognisably brown) — held-out tests robustness to shade/lighting/pose/position, not a change
  of what the colours mean.
- The agent should also produce a one-line `judge/README.md` stating how to run the held-out eval and
  reaffirming it must not be shared with competitors.

### Optional sanity check (agent may run, do not over-train)

Running `eval.py` with `judge/heldout.yaml` on a lightly-trained hard checkpoint should execute end to
end and produce the metrics block — confirming the held-out path is wired identically to the visible
one. Expect low scores from an undertrained model; the point is pipeline parity, not performance.

---

## 11. Smoke test before declaring done

The scaffold is "done" when all of the following pass (these are also what the human verifier will
check, so make them runnable):

1. `python train.py difficulty=easy total_steps=<small>` runs end-to-end and produces a checkpoint.
2. The trained easy policy sorts at least one parcel under `test.py` (loop closes; not a fully
   trained model).
3. `python eval.py difficulty=easy checkpoint=<path> eval_config=conf/eval/default.yaml` runs and
   prints the §9.1 metrics summary.
4. `difficulty=medium` constructs and returns a wrist-camera RGB observation of the documented shape;
   the tag is visible in the rendered image.
5. `difficulty=hard` constructs with all randomisation axes active at narrow ranges; rendering a few
   episodes visibly shows varied bin sides, lighting, and backgrounds, with tags still top-visible.
6. `python eval.py difficulty=hard checkpoint=<path> eval_config=judge/heldout.yaml` runs end to end
   through the identical pipeline and prints the §9.1 metrics block (low scores from an undertrained
   model are fine — this proves the held-out path is wired identically to the visible one).
7. `reward=example_dense difficulty=easy` trains and reaches higher sort accuracy than `reward=sparse`
   in a short run (confirms the example reward is wired and functional).
8. `notebook.ipynb` executes top-to-bottom in a fresh runtime (see §12a), finishing on a visible
   trained-policy rollout.

Do not over-train in the smoke test — just verify the interfaces, the loop, and that the rendered
scenes look right.

---

## 12. Build order

1. Easy env (state obs): scene (tabletop + brown parcels with coloured top tags + low colour-coded
   bins L/R) + action space + `evaluate` (§7 metrics) + sparse reward. `train.py` closes the loop.
2. Add the example dense reward (§6) behind `reward=example_dense`; verify it beats sparse on easy.
3. `test.py` (same-distribution self-check, §9.1 metrics) and `eval.py` (config-driven, obs-mode
   locked, with visible `conf/eval/default.yaml`). Aggregation reproducible from seeds.
4. Define the §8 randomisation schema once; wire `randomization/train.yaml` (narrow ranges).
5. Medium: `obs_mode: rgb` (wrist cam) + `parcel_pose` randomisation, tag top-visible — config only.
6. Hard: light clutter + ALL randomisation axes at narrow training ranges (still 2 colours/2 bins) — config only.
7. `judge/heldout.yaml` (concrete widened/recombined ranges, real values) + `judge/README.md`, with
   `judge/` gitignored and excluded from the competitor-facing repo (§10).
8. README (§ participant-facing content), with all `<agent: fill in>` placeholders populated.
9. `notebook.ipynb` (§12a) — the guided step-by-step runner. Build this last, after the scripts and
   configs work, and verify it runs top-to-bottom in a fresh runtime.

Stop cleanly after whichever level is reached if time runs short — each level is independently
runnable. Build in this order so each step is independently verifiable.

---

## 12a. Colab notebook (`notebook.ipynb`)

A guided, step-by-step notebook usable in Google Colab. It is **participant-facing** and is a thin
runner over the real code — **not** a reimplementation.

Requirements:

- **Setup cells first:** install ManiSkill and dependencies, install the package (`pip install -e .`),
  and a markdown note to select a GPU runtime. Handle Colab headless rendering (offscreen GL) so
  render calls work in a Colab VM.
- **Guided pipeline, ordered runnable cells with markdown between them:**
  1. Construct the env; render a few frames of each difficulty (show brown parcels with coloured tags,
     coloured L/R bins, and — for hard — visible randomisation).
  2. Inspect observations: display a wrist-camera RGB frame and the state vector, with their documented
     shapes/dtypes.
  3. Short `train.py` run (small `total_steps`, modest `num_envs` so it completes on a Colab GPU);
     clearly mark where a user would scale up for real training.
  4. `test.py` on the trained checkpoint.
  5. `eval.py` on `conf/eval/default.yaml`, printing the §9.1 metrics.
  6. Render at least one trained-policy rollout so the notebook ends on a visible result.
- **Call into the real package and configs** — cells invoke the actual scripts / env / Hydra configs,
  never duplicate env, training, or reward logic. The notebook must stay consistent with the code by
  construction.
- **Never reference `judge/`** or the held-out config. The notebook uses only `conf/` configs.
- **Verify:** the notebook must execute top-to-bottom in a fresh runtime before the build is declared
  done.

---

## 13. Deliverables

- The package (one env file, config tree, three scripts: `train.py`, `test.py`, `eval.py`).
- The §8 randomisation schema + `conf/randomization/train.yaml` (narrow training ranges).
- `conf/eval/default.yaml` (visible, same-distribution).
- `judge/heldout.yaml` (concrete held-out config, real widened values) + `judge/README.md`, with
  `judge/` **gitignored and excluded from the competitor-facing repo** — delivered separately to judges.
- Sparse reward (default) + example dense reward behind `reward=example_dense`, clearly marked.
- A **README.md** containing the participant-facing content in the next section.
- `notebook.ipynb` (§12a) — guided step-by-step runner, verified to run top-to-bottom, competitor-facing
  (no `judge/` reference).
- All `<agent: fill in>` placeholders in the README populated from the actual implementation
  (observation shapes/dtypes/channel order, state-vector layout, action dimensions, install command,
  exact geometric success check, colour→bin palette).
