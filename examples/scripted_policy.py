"""Scripted pick-and-place policy for WarehouseSortEnv.

A deterministic motion-primitive state machine that solves the task without any learning.
Purpose: prove the environment is physically solvable before spending GPU time on RL.

Phases (per parcel, in order):
  OPEN     → open gripper
  ABOVE    → move TCP directly above parcel at safe height
  DESCEND  → lower onto parcel top face
  GRASP    → close gripper
  LIFT     → raise to carry height (clears bin walls)
  CARRY    → move laterally to above correct bin
  DROP     → lower into bin
  RELEASE  → open gripper, retreat

Actions: pd_ee_delta_pos, 4-dim, range [-1, 1].
  dims 0-2: delta xyz of end-effector (action * 0.1 m/step)
  dim  3  : gripper (+1 = open, -1 = close)

Usage:
  pixi run python examples/scripted_policy.py
  # video saved to outputs/scripted/videos/
"""

import os
import gymnasium as gym
import torch
import warehouse_sort  # noqa — registers WarehouseSort-v1

from mani_skill.utils.wrappers.record import RecordEpisode

# ── geometry constants ─────────────────────────────────────────────────────── #
SCALE  = 0.1     # metres per unit action (controller pos_upper)
HOVER   = 0.22   # safe transit height above table (clears parcels)
GRASP_Z = 0.060  # TCP z at parcel top-face level (fingertips at ~0.061, grip upper box sides)
CARRY   = 0.26   # carry height — lift high so the box clears all parcels/bin walls when moving
DROP_Z  = 0.08   # TCP z inside bin (parcel settles ~0.031)
SPEED   = 0.7    # max action magnitude per step (fraction of 0.1m) — 0.7 → 7 cm/step
TOL     = 0.015  # "close enough" threshold (m) before advancing phase

# ── phase codes ────────────────────────────────────────────────────────────── #
OPEN, ABOVE, DESCEND, GRASP, LIFT, CARRY_P, DROP, RELEASE = range(8)
PHASE_NAMES = ["OPEN", "ABOVE", "DESCEND", "GRASP", "LIFT", "CARRY", "DROP", "RELEASE"]


def _act(delta_xyz, gripper):
    """Build a (1, 4) action tensor from a numpy array and scalar gripper."""
    dx, dy, dz = (float(v) for v in delta_xyz)
    g = float(gripper)
    return torch.tensor([[dx, dy, dz, g]], dtype=torch.float32)


def _move(tcp, target, gripper=1.0, speed=SPEED):
    """Proportional-control step toward target; clamped to [-speed, speed]."""
    import numpy as np
    delta = (target - tcp).clip(-speed, speed)
    return _act(delta / SCALE, gripper)


def _at(tcp, target, tol=TOL):
    import numpy as np
    return float(((tcp - target) ** 2).sum() ** 0.5) < tol


def scripted_episode(env, max_steps=300):
    """Run one scripted episode on a single-env WarehouseSortEnv wrapper.

    Returns a list of (obs, action, reward, info) tuples.
    """
    import numpy as np

    base = env.unwrapped
    obs, _ = env.reset(seed=42)
    device = "cpu"

    phase       = OPEN
    parcel_idx  = 0
    phase_steps = 0       # steps spent in the current phase (reset on every transition)
    grasp_tries = 0       # grasp attempts on the current parcel (give up after a few)
    n_parcels   = base.num_parcels
    history     = []

    def goto(p):
        nonlocal phase, phase_steps
        phase, phase_steps = p, 0

    # Spread drop points within a bin so multiple same-colour parcels don't stack:
    # each parcel gets a "slot" index among parcels sharing its tag, offset along the
    # bin's x-axis (footprint half-x = 0.11, so +/-0.075 keeps boxes inside).
    tags0 = base.parcel_tags[0].cpu().long().tolist()
    slot, per_tag = [0] * n_parcels, {}
    for j in range(n_parcels):
        t = tags0[j]
        slot[j] = per_tag.get(t, 0)
        per_tag[t] = slot[j] + 1
    tag_total = dict(per_tag)

    for step in range(max_steps):
        phase_steps += 1
        tcp  = base.agent.tcp_pose.p[0].cpu().numpy()
        bins = base._bin_positions()[0].cpu().numpy()     # (2, 3)
        tags = base.parcel_tags[0].cpu().long().tolist()  # [tag_p0, tag_p1, ...]

        if parcel_idx >= n_parcels:
            action = _act([0, 0, 0], 1.0)
        else:
            p_pos = base.parcels[parcel_idx].pose.p[0].cpu().numpy()
            tag   = tags[parcel_idx]
            bin_xyz = bins[tag]    # correct bin for this parcel
            # slot offset within the bin, spread along the bin's deeper y-axis (footprint
            # half-y = 0.13, vs half-x = 0.11) so 3 same-colour boxes fit without one landing
            # on the rim. +/-0.07 keeps each box (half 0.026) well inside the footprint.
            off = (slot[parcel_idx] - (tag_total[tag] - 1) / 2.0) * 0.07

            above_p  = np.array([p_pos[0],   p_pos[1],          HOVER])
            grasp_p  = np.array([p_pos[0],   p_pos[1],          GRASP_Z])
            carry_p  = np.array([bin_xyz[0], bin_xyz[1] + off,  CARRY])
            drop_p   = np.array([bin_xyz[0], bin_xyz[1] + off,  DROP_Z])

            def advance_parcel():
                nonlocal parcel_idx, grasp_tries
                parcel_idx += 1
                grasp_tries = 0
                goto(ABOVE if parcel_idx < n_parcels else OPEN)

            if phase == OPEN:
                action = _act([0, 0, 0], 1.0)
                if phase_steps >= 8:
                    goto(ABOVE)

            elif phase == ABOVE:
                # Align VERY tightly in xy at hover height before descending. With a ~5cm box
                # rotated by up to 0.5 rad, the open gripper has only ~5mm clearance per side,
                # so the lateral error must be small or a finger clips the box on the way down.
                action = _move(tcp, above_p, gripper=1.0, speed=0.4)
                lat = float(np.linalg.norm(tcp[:2] - above_p[:2]))
                if (lat < 0.005 and abs(tcp[2] - HOVER) < 0.05) or phase_steps > 70:
                    goto(DESCEND)

            elif phase == DESCEND:
                # Slow, near-vertical descent (xy already aligned). Exit on low z with xy still
                # tight; keying on z (not a step count) stops it grasping too high.
                action = _move(tcp, grasp_p, gripper=1.0, speed=0.12)
                lat = float(np.linalg.norm(tcp[:2] - grasp_p[:2]))
                if (tcp[2] <= GRASP_Z + 0.008 and lat < 0.008) or phase_steps > 60:
                    goto(GRASP)

            elif phase == GRASP:
                action = _act([0, 0, 0], -1.0)
                if phase_steps >= 12:
                    if base.agent.is_grasping(base.parcels[parcel_idx])[0].item():
                        goto(LIFT)
                    else:
                        grasp_tries += 1
                        goto(ABOVE if grasp_tries < 3 else RELEASE)  # give up → skip parcel

            elif phase == LIFT:
                # rise STRAIGHT UP (no lateral motion) so the carried box clears neighbouring
                # parcels and bin walls before the lateral carry begins.
                action = _act([0, 0, 1.0], -1.0)
                if tcp[2] > CARRY - 0.01 or phase_steps > 35:
                    goto(CARRY_P)

            elif phase == CARRY_P:
                action = _move(tcp, carry_p, gripper=-1.0)
                if _at(tcp, carry_p, tol=0.04) or phase_steps > 40:
                    goto(DROP)

            elif phase == DROP:
                action = _move(tcp, drop_p, gripper=-1.0, speed=0.3)
                if _at(tcp, drop_p, tol=0.025) or phase_steps > 30:
                    goto(RELEASE)

            elif phase == RELEASE:
                action = _act([0, 0, 0.1], 1.0)   # open + small lift
                if phase_steps >= 10:
                    advance_parcel()

        obs, reward, term, trunc, info = env.step(action.to(device))
        history.append((obs, action, float(reward), info))

        sc = info.get("success_count", None)
        if sc is not None:
            sc_val = sc.item() if hasattr(sc, "item") else sc
        else:
            sc_val = "?"
        if step % 20 == 0 or step == max_steps - 1:
            print(f"  step {step:3d}  phase={PHASE_NAMES[phase] if parcel_idx < n_parcels else 'DONE':8s}"
                  f"  parcel={parcel_idx}  tcp={tcp.round(3)}  sorted={sc_val}", flush=True)

        if term or trunc:
            break

    return history


# Difficulty → WarehouseSortEnv kwargs (mirrors conf/difficulty/*.yaml):
#   easy   = 2 parcels (1 each colour), fixed poses, no randomisation
#   medium = 4 parcels (2 each colour), randomised poses, bins fixed
#   hard   = 6 parcels (3 each colour), full randomisation, bins SWAPPED for this demo
_RAND_MEDIUM = {
    "parcel_pose":  {"xy_jitter": [-0.03, 0.03], "yaw_jitter": [-0.5, 0.5]},
    "bin_position": {"side_swap_prob": 0.0, "xy_jitter": [0.0, 0.0]},
}

# Hard demo: force the red/blue bins to swap sides (side_swap_prob=1.0) so the video
# visibly shows the policy routing by tag colour, not by a memorised side. The scripted
# policy reads bin positions live, so it sorts correctly regardless.
_RAND_HARD = {
    "parcel_pose":  {"xy_jitter": [-0.03, 0.03], "yaw_jitter": [-0.5, 0.5]},
    "bin_position": {"side_swap_prob": 1.0, "xy_jitter": [-0.02, 0.02]},
    "lighting":     {"intensity": [0.8, 1.2], "direction_jitter": [-0.2, 0.2]},
    "background":   {"table_colors": [[0.32, 0.30, 0.28], [0.28, 0.30, 0.30]]},
    "appearance":   {"cardboard_shade": [-0.06, 0.06], "tag_shade": [-0.05, 0.05]},
}

DIFFICULTY_KWARGS = {
    "easy":   dict(num_parcels=2, fixed_poses=True,  randomization=None),
    "medium": dict(num_parcels=4, fixed_poses=False, randomization=_RAND_MEDIUM),
    "hard":   dict(num_parcels=6, fixed_poses=False, randomization=_RAND_HARD),
}


def run_difficulty(difficulty: str, seed: int = 42):
    out_dir = f"outputs/scripted/{difficulty}"
    os.makedirs(out_dir, exist_ok=True)

    kwargs = DIFFICULTY_KWARGS.get(difficulty, DIFFICULTY_KWARGS["easy"])
    n_parcels = kwargs["num_parcels"]
    # budget ~100 steps per parcel for the full lift-high pick-carry-place cycle (+ retry)
    max_steps = max(150, 100 * n_parcels)
    env = gym.make(
        "WarehouseSort-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        sim_backend="gpu",
        render_mode="all",
        max_episode_steps=max_steps,
        **kwargs,
    )
    env = RecordEpisode(
        env,
        output_dir=out_dir,
        save_trajectory=False,
        save_video=True,
        video_fps=20,
        max_steps_per_video=max_steps,
    )

    print(f"\n=== scripted policy  difficulty={difficulty}  "
          f"({n_parcels} parcels)  seed={seed} ===")
    history = scripted_episode(env, max_steps=max_steps)
    env.close()

    final_info = history[-1][-1]
    sc = final_info.get("success_count")
    sc_val = sc.item() if hasattr(sc, "item") else sc
    print(f"Final sorted: {sc_val} / {n_parcels}")
    print(f"Video → {out_dir}/0.mp4")
    return sc_val


def main():
    import sys
    difficulties = sys.argv[1:] if len(sys.argv) > 1 else ["easy", "medium", "hard"]
    for d in difficulties:
        run_difficulty(d)


if __name__ == "__main__":
    main()
