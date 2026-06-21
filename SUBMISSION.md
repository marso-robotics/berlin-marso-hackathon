# Submission Guide — WarehouseSort

This is the one-page map of how the pieces fit together, how to reproduce the
provided Diffusion-Policy solution, how to bring your own approach, and how to
package what you submit. For the full task spec, observation/action layout, and
scoring weights, see the [README](README.md).

---

## 0. How to submit

You submit a **GitHub repository** (a fork of this repo, or your own repo built on
it) — not just a checkpoint. We clone it and run your policy as-is.

Your repo must contain:

1. **Your policy entrypoint** — a `module:function` that loads your checkpoint and
   returns a policy (the provided `warehouse_sort.il_policy:load_dp` already works;
   for a custom approach it's your own module — see §3).
2. **Your checkpoint file(s)** — committed to the repo (or fetchable by a script in
   it). Reference them by path in the manifest.
3. **`submission.yaml`** at the repo root — declares the track(s) you trained (`state`,
   `rgb`, or both), each with its policy entrypoint + a checkpoint per level (see §4).

Make sure a fresh clone installs cleanly (`pixi install && pixi run install`) and
that your entrypoint imports without your training code present — evaluation only
ever calls your `load_fn` and `.act(obs)`.

---

## 1. How the code ties together

```
              you train                     you declare           evaluated on
  ┌─────────────────────────────┐   ┌────────────────────────┐   held-out configs
  demos ─► il/train.py ─► ckpt.pt    submission.yaml           ┌───────────────────┐
                  │                    state/rgb tracks:      │   eval.py pipeline │
                  │                     policy + per-level ckpt└─────────┬─────────┘
                  ▼                          │                            │
       warehouse_sort/il_policy.py ◄─────────┘                            ▼
       load_dp(ckpt, obs, act, dev)                                  sort_accuracy
                  │                                                       ▲
                  ▼                                                       │
       policy.act(obs, deterministic=True) ─► action in [-1,1] ──────────┘
```

The single contract that glues everything is the **policy entrypoint**:

```python
load_fn(checkpoint, sample_obs, action_space, device) -> policy
policy.act(obs, deterministic=True) -> Tensor (num_envs, action_dim) in [-1, 1]
```

`eval.py` never imports your *training* code. It calls your `load_fn` (given by
`policy=module:function`), hands it the checkpoint path + a sample observation + the
action space, then steps the env calling `.act(obs)`. That's the whole interface. See
[warehouse_sort/utils.py](warehouse_sort/utils.py) `load_agent` (L115) and
`rollout_metrics` (L64).

> **Why we need your code, not just a checkpoint.** Evaluation resolves
> `policy=module:function` with `importlib.import_module` — so your `load_fn` (and any
> model class it builds) must be importable from your repo. The checkpoint format is
> entirely yours: only your `load_fn` reads it, which is exactly why the code has to
> ship with it. Evaluation only ever calls `.act(obs)`, so your code can't touch
> privileged env state.

Key files:

| File | Role |
|------|------|
| [warehouse_sort/env.py](warehouse_sort/env.py) | The `WarehouseSort-v1` env: scene, obs, sparse reward, `evaluate()` (success check) |
| [warehouse_sort/il_policy.py](warehouse_sort/il_policy.py) | `load_dp` — reference policy entrypoint (state; `load_dp_rgb` for the image track) |
| [warehouse_sort/utils.py](warehouse_sort/utils.py) | env construction, deterministic rollout, metrics |
| [eval.py](eval.py) | evaluate a checkpoint on an eval config (**the exact interface used for scoring** — only the config differs) |
| [conf/](conf/) | Hydra configs: `difficulty/{easy,medium,hard}.yaml`, `eval/default.yaml` |

---

## 2. Run the provided solution (state Diffusion Policy)

This is the full state-IL pipeline, runnable end-to-end.
Full detail in [il/README.md](il/README.md); the short version (the demos — 200 episodes per
level — are the Kaggle competition data; mounted automatically on Kaggle, else fetched below):

```bash
pixi install && pixi run install

# fetch the demo datasets (the Kaggle competition data) -> il/demos/<level>/
pixi run python il/download_demos.py

# train the state Diffusion Policy — ONE checkpoint PER level (state is parcel-count-specific)
pixi run python il/train.py method=dp demo_dir=easy   # then demo_dir=medium, demo_dir=hard
```

> ⚠️ **Your submission must be a *learned* policy** (a parameterized model trained to map the
> observation to actions). The provided demos come from a scripted controller, and using it to
> collect *more* data is fine — but **submitting** a scripted / hard-coded / rule-based controller
> (even one that only reads the provided observation), or anything that reads privileged simulator
> state, is **not a valid submission** and leads to disqualification.

Check progress any time with `eval.py` — the **exact same interface used for scoring**.
You get `conf/eval/default.yaml` (same-distribution seeds); scoring swaps in a held-out
config (wider ranges, different seeds), but the pipeline is identical:

```bash
pixi run python eval.py difficulty=easy \
    policy=warehouse_sort.il_policy:load_dp \
    checkpoint=il/baselines/diffusion_policy/runs/warehouse_state_dp_easy/checkpoints/best_eval_sort_accuracy.pt \
    eval_config=conf/eval/default.yaml
```

To **continue improving**: train longer (`flags.total_iters=`), tune the prediction horizon
(`flags.pred_horizon=`), record extra demos (optional — see [il/README.md](il/README.md)), or
improve generalization to the held-out positions / bin-swaps (hard is weighted 0.5).

---

## 3. Bring your own approach (any policy)

You do **not** have to use Diffusion Policy — any **learned** policy satisfying the contract works
(behavior cloning, RL, a transformer, …). It just must be a trained observation→action model, not
a hand-coded controller. The minimal reference for the *contract* (not a valid submission — it's
random) is [examples/random_policy.py](examples/random_policy.py):

```python
class RandomPolicy:
    def __init__(self, action_space, device):
        self.action_dim = action_space.shape[0]
        self.device = device

    def act(self, obs, deterministic=True):
        # obs is the state vector (N, obs_dim) for the main track, or a dict
        # {"rgb": (N,128,128,3), "state": (N,26)} for the image track. Read ONLY the obs.
        n = (obs["rgb"] if isinstance(obs, dict) else obs).shape[0]
        return torch.rand((n, self.action_dim), device=self.device) * 2 - 1

def load_policy(checkpoint, sample_obs, action_space, device):
    return RandomPolicy(action_space, device)
```

Two rules your `load_fn` must follow (copy them from `il_policy.py:load_dp`):

1. **Build the model from `sample_obs` + `action_space`** — size the network off `sample_obs`
   (the state vector's dim depends on parcel count) rather than hardcoding shapes.
2. **Load the checkpoint yourself** from the `checkpoint` path argument and move the
   model to `device`. `load_dp` does exactly this:
   ```python
   ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
   sd = ckpt.get("ema_agent", ckpt.get("agent"))   # your own key/format is fine
   net.load_state_dict(...)
   ```
   Only the path string is passed in — what's inside the file is entirely your format.
   Just make sure `load_fn(path, ...)` reconstructs a ready-to-run policy with no code
   changes on the evaluation side.

Then point the manifest at it: `policy: my_module:load_policy`.

---

## 4. The submission manifest

Submit **one `submission.yaml`** at your repo root (template:
[submission.example.yaml](submission.example.yaml)). Declare the **track(s)** you trained —
`state`, `rgb`, or **both**. Each track has its own policy entrypoint and one checkpoint per
difficulty level:

```yaml
team: "team-name"

state:                                       # main track (privileged state vector)
  policy: warehouse_sort.il_policy:load_dp
  levels:
    easy:   { checkpoint: <path-to-easy-ckpt> }
    medium: { checkpoint: <path-to-medium-ckpt> }
    hard:   { checkpoint: <path-to-hard-ckpt> }

rgb:                                         # optional image track — omit if you didn't do it
  policy: warehouse_sort.il_policy:load_dp_rgb
  levels:
    easy:   { checkpoint: <path-to-easy-ckpt> }
    medium: { checkpoint: <path-to-medium-ckpt> }
    hard:   { checkpoint: <path-to-hard-ckpt> }
```

- **Tracks** — include `state`, `rgb`, or both. Each is scored independently; the judge runs the
  env in that observation mode (state vs rgb) so your policy gets the inputs it expects.
- **policy** — the `module:function` entrypoint for that track (`load_dp` for state, `load_dp_rgb`
  for rgb, or your own).
- **levels / checkpoint** — one checkpoint per difficulty (the state vector is
  parcel-count-specific). Omit a level → it scores 0, but its weight still counts.

## Scoring

The primary metric is **sort accuracy** — the fraction of parcels placed in the correct-color
bin. Each declared level is scored on a held-out config, then combined **per track**:

```
final = 0.2 · sort_accuracy_easy + 0.3 · sort_accuracy_medium + 0.5 · sort_accuracy_hard
```

The judge reports a weighted score for **each track you submitted** (e.g. a state score and an
rgb score). Higher weight on harder levels rewards generalization.
