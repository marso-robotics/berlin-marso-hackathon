# Imitation-Learning track — WarehouseSort-v1

The IL path, parallel to the RL/reward track. It follows ManiSkill 3's own IL pipeline —
**record demos → `replay_trajectory` into the training representation → train a vendored
ManiSkill baseline → evaluate through the project's `eval.py`** — and never hand-rolls. The
scripted waypoint policy in `examples/scripted_policy.py` is the demo source (we reuse its
grasp/place logic).

**Diffusion Policy (DP) is the working template.** A plain MLP behaviour-cloner collapses here
(compounding error on a fixed, single-path dataset → ~0% success); DP's action chunking fixes
that. Two variants:
- **DP (state)** — privileged low-dim state, the easy plumbing baseline. **85% sort accuracy.**
- **DP (rgb, scene cam)** — image + proprioception only, the real/generalisable policy.

---

## 0. Environment facts (confirmed empirically on this build)

| thing | value |
|-------|-------|
| ManiSkill | 3.0.1 (pip wheel — does **not** ship `examples/baselines`, so we vendor them, §3) |
| action | `Box(-1,1,(4,))` — `pd_ee_delta_pos`: `[dx, dy, dz, gripper]` (+1 open / −1 close) |
| state obs (easy) | flat `float32 (N, 54)` — privileged: parcel poses+tags, bin pos+colours, proprio |
| **rgb obs** | `obs_camera=scene` (default): a fixed third-person camera. `FlattenRGBDObservationWrapper(rgb=True, depth=False, state=True)` → `{"rgb": (N,128,128,3) uint8, "state": (N,26) f32}` |
| rgb `state` (26-d) | **proprioception only** — `qpos(9)+qvel(9)+tcp_pose(7)+is_grasped(1)`. **No** parcel/bin/tag. **No depth.** |
| `obs_camera` | `scene` (default — whole table/bins/robot always visible, robot=`panda`) \| `wrist` (Panda wrist cam, robot=`panda_wristcam`) |
| eval horizon | IL uses `max_episode_steps=200` (config default) — room for the 2-parcel sort + clean return-home |

Why the scene camera: the wrist camera is occluded by a parcel once grasped, so wrist-cam DP
never learns to place (verified 0%). A fixed third-person view keeps the bins and robot in
frame the whole episode and is parcel-count-agnostic, so the **same** rgb policy can be
evaluated on any difficulty.

---

## 1. Generate demonstrations

```bash
pixi run python il/gen_demos.py --num-episodes 60 --action-noise 0.05   # obs_camera defaults to scene
```

1. Rolls the scripted policy across 60 seeds on easy, recording ManiSkill `.h5`+`.json` via
   `RecordEpisode`. Each demo plays out a **clean finish** (release → lift → return home, hold)
   so every trajectory ends in an unambiguous parked state.
2. **Demo-time action noise** (`--action-noise 0.05`, ~5 mm/step on xyz): the scripted policy is
   closed-loop and self-corrects, widening the demo state distribution (standard covariate-shift fix).
3. Runs ManiSkill's **`replay_trajectory`** (via `il/_replay.py`, which registers our env first) to
   produce `trajectory.state.*.h5` and `trajectory.rgb.*.h5` (rgb = the scene camera).

> Version notes: this wheel's `replay_trajectory` uses `--num-envs` (not `--num-procs`);
> control-mode conversion is GPU-unsupported (we don't convert — replay only re-renders obs);
> set `HDF5_USE_FILE_LOCKING=FALSE` if replay/load races on the just-written `.h5`.

---

## 2. Train (thin runner over the vendored baselines)

```bash
pixi run python il/train.py dp        # Diffusion Policy, STATE  (easy plumbing baseline)
pixi run python il/train.py dp_rgb    # Diffusion Policy, RGB scene cam  (the real template)
# also available: bc (MLP, fails), act (state, tuning-sensitive)
```

`il/train.py` invokes the real vendored scripts with the demo path, control mode, scene camera,
and the 200-step horizon. Checkpoints land under `il/baselines/<dir>/runs/<exp>/checkpoints/`.

## 3. Vendored ManiSkill baselines

The wheel omits `examples/baselines`, so `bc`, `act`, `diffusion_policy` are copied into
`il/baselines/` with a one-line shim in each `make_env.py` (`import warehouse_sort`). The only
authored change is `train_rgbd.py`'s `obs_camera` arg (wrist→scene). Extra deps: `diffusers`
(DP/ACT), `torchvision` (ACT only).

## 4. Evaluate through the judges' `eval.py`

Each method has a policy entrypoint in `warehouse_sort/il_policy.py` satisfying the contract
(`act(obs, deterministic) -> action in [-1,1]`); DP is deployed **fully closed-loop** (re-query
each step, execute the first predicted action — action horizon 1 ≤ prediction horizon).

```bash
# DP state (easy)
pixi run python eval.py difficulty=easy obs_mode=state \
    policy=warehouse_sort.il_policy:load_dp \
    checkpoint=il/baselines/diffusion_policy/runs/warehouse_state_dp_v2/checkpoints/best_eval_success_at_end.pt \
    eval_config=conf/eval/default.yaml

# DP rgb (scene cam) — same checkpoint works across difficulties (fixed image input)
pixi run python eval.py difficulty=easy obs_mode=rgb obs_camera=scene \
    policy=warehouse_sort.il_policy:load_dp_rgb \
    checkpoint=il/baselines/diffusion_policy/runs/warehouse_rgb_dp/checkpoints/best_eval_success_at_end.pt \
    eval_config=conf/eval/default.yaml
```

Entrypoints: `load_dp` (state), `load_dp_rgb` (rgb scene). `load_bc` also exists.

### OOD generalisation (rgb scene policy only)

Because the image input shape is fixed, the easy-trained rgb DP can be *run* on harder configs:
`difficulty=medium`, `difficulty=hard`, and a same-shape **bins-swapped** test
(`eval_config=conf/eval/easy_swapped.yaml`). (The state policy is parcel-count-specific, so it
cannot be fed medium/hard.)

## 5. Results (easy)

| method | obs | extra deps | eval (sort_accuracy) |
|--------|-----|-----------|----------------------|
| MLP-BC | state | none | ~0% (compounding error) |
| **DP** | **state** | `diffusers` | **0.85** (1.70/2, 0% mis-sort) |
| DP | rgb (scene) | `diffusers` | _filled in after training_ |
| ACT | state | `diffusers`, `torchvision` | deferred (tuning-sensitive) |

## 6. Notes
- **Colab headless** — offscreen Vulkan works on GPU runtimes; render few envs (rgb is the cost).
- Never reference `judge/` from this track.
