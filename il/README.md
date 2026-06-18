# Imitation Learning — WarehouseSort

The IL pipeline follows ManiSkill 3's standard approach:
**record demos → replay_trajectory → train Diffusion Policy → evaluate via eval.py**.

**State DP works (~85% sort accuracy on easy).** RGB/image IL is provided as a template but is
not yet solving the task.

---

## Step 1 — Generate demonstrations

```bash
pixi run python il/gen_demos.py --num-episodes 60 --action-noise 0.05
```

This rolls the scripted waypoint policy across 60 seeds on easy (2 parcels, fixed layout),
records raw trajectories via ManiSkill's `RecordEpisode`, then runs `replay_trajectory` to
produce training-ready datasets:
- `il/demos/easy/trajectory.state.*.h5` — privileged state
- `il/demos/easy/trajectory.rgb.*.h5` — scene-camera RGB

Action noise (`--action-noise 0.05`, ~5 mm/step on xyz) spreads the demo state distribution
to reduce behavior cloning covariate shift. The policy is closed-loop and self-corrects.

To generate for other levels:
```bash
pixi run python il/gen_demos.py --difficulty medium --num-episodes 100 --action-noise 0.05
pixi run python il/gen_demos.py --difficulty hard   --num-episodes 150 --action-noise 0.05
```

---

## Step 2 — Train

```bash
pixi run python il/train.py method=dp               # state Diffusion Policy (recommended)
pixi run python il/train.py method=dp_rgb           # RGB Diffusion Policy (template only)
```

To train on a different level pass `demo_dir=<level>`:
```bash
pixi run python il/train.py method=dp demo_dir=medium
```

Override any hyperparameter on the CLI:
```bash
pixi run python il/train.py method=dp flags.total_iters=50000 flags.eval_freq=5000
```

Checkpoints land at `il/baselines/diffusion_policy/runs/<exp_name>/checkpoints/`.

---

## Step 3 — Evaluate

```bash
# State DP on easy
pixi run python eval.py difficulty=easy obs_mode=state \
    policy=warehouse_sort.il_policy:load_dp \
    checkpoint=il/baselines/diffusion_policy/runs/warehouse_state_dp_easy/checkpoints/best_eval_success_at_end.pt \
    eval_config=conf/eval/default.yaml

# RGB DP on easy (template — low accuracy expected)
pixi run python eval.py difficulty=easy obs_mode=rgb obs_camera=scene \
    policy=warehouse_sort.il_policy:load_dp_rgb \
    checkpoint=<rgb_checkpoint> \
    eval_config=conf/eval/default.yaml
```

`eval.py` locks obs_mode to the difficulty default. For the main (state) track obs_mode=state
is always used; for the image track obs_mode=rgb.

---

## Results

| method | obs | level | sort_accuracy |
|--------|-----|-------|--------------|
| **DP** | **state** | **easy** | **~0.85** |
| DP | state | medium | _run to fill_ |
| DP | state | hard | _run to fill_ |
| DP | rgb (scene) | easy | template only |

---

## Technical notes

- **Why state DP and not MLP-BC?** A plain MLP behavior cloner collapses here (~0% success)
  due to compounding error. Diffusion Policy's action chunking fixes this.
- **Image input = a single fixed third-person scene camera.** It keeps the whole workspace
  (robot + parcels + bins) in frame the entire episode and works at any parcel count.
- ManiSkill 3.0.1 pip wheel does not ship `examples/baselines`, so the DP baseline is
  vendored in `il/baselines/diffusion_policy/`.
- Set `HDF5_USE_FILE_LOCKING=FALSE` if replay/load races on the just-written `.h5`.
