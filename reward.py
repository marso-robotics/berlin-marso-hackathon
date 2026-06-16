"""warehouse_sort/reward.py — staged, wall-aware dense reward for WarehouseSort-v1.

Design principle
----------------
The task is a *sequence* of pick-and-place sub-goals, each of which requires lifting a
parcel over a walled bin before dropping it inside.  A naive PickCube-style "distance to
goal" potential fails here because:

  1. The straight-line gradient to the bin floor passes *through* the bin wall — the policy
     learns to wedge the box against the rim and stall.
  2. Summing per-parcel potentials creates competing gradients (grasped vs un-grasped).
  3. The sparse "settled inside bin" event almost never fires to bootstrap the place phase.

Fix: a 4-phase state machine per parcel, with a *clearance waypoint* above the bin rim
between the carry and drop phases, plus explicit sequencing (reward only the current
target parcel at any moment).

Phases (per parcel, in order)
------------------------------
  0  IDLE       — already placed correctly, contribute 0 (done bonus already given)
  1  REACH      — arm TCP moves toward the parcel
  2  CARRY      — parcel grasped, TCP moves to waypoint *above* the target bin rim
  3  DROP       — parcel is above the bin interior, lower it in

Each phase is converted to a value ∈ [0, 1] via tanh potentials.  Bonuses (+fixed) fire
at phase transitions to make exploration over the whole episode worthwhile.

Hyperparameters are intentionally visible at the top so the agent building on this can
tune them without touching the logic.
"""

from __future__ import annotations

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (all in SI / normalised units)
# ──────────────────────────────────────────────────────────────────────────────

# Spatial sharpness of tanh potentials.  Larger k → sharper, harder to optimise early.
K_REACH = 4.0    # reach-to-parcel
K_CARRY = 3.0    # carry-to-waypoint (above bin rim)
K_DROP  = 5.0    # drop-to-bin-floor

# Height of the clearance waypoint above the table surface (metres).
# Must clear the bin rim (full wall height = 2 * bin_wall_h = 0.05 m) with margin.
WAYPOINT_Z = 0.12

# Lateral tolerance: parcel must be within this radius of the bin center when switching
# from CARRY to DROP phase (so it really is above the bin, not just high).
LATERAL_TOL = 0.07   # metres; bin footprint half-extents are (0.11, 0.13)

# z threshold below which a parcel is considered "settled" (same as evaluate()).
SETTLED_Z = 0.06

# Fixed bonuses at each phase transition (additive, not scaled by num_parcels).
BONUS_GRASP   = 0.5   # fired once when a parcel is newly grasped
BONUS_ABOVE   = 1.0   # fired once when parcel clears the rim above the correct bin
BONUS_PLACE   = 3.0   # fired once when parcel settles inside the correct bin

# Scale factor applied to the continuous potential terms before summing with bonuses.
# Keeps the per-step signal in a similar range to the bonuses.
POT_SCALE = 0.5

# ──────────────────────────────────────────────────────────────────────────────
# Maximum possible reward for one episode (used by compute_normalized_dense_reward)
# ──────────────────────────────────────────────────────────────────────────────

def example_dense_max(num_parcels: int) -> float:
    """Upper bound on reward.  Used to normalise to [0, 1].

    Per parcel: POT_SCALE*(reach+carry+drop each capped at 1) + three bonuses.
    Plus the global all-placed bonus counted once.
    """
    per_parcel = POT_SCALE * 3.0 + BONUS_GRASP + BONUS_ABOVE + BONUS_PLACE
    return float(num_parcels) * per_parcel + BONUS_PLACE  # last BONUS_PLACE as all-placed bonus


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tanh_pot(dist: torch.Tensor, k: float) -> torch.Tensor:
    """Smooth potential ∈ (0, 1]: 1 when dist==0, decays with k."""
    return 1.0 - torch.tanh(k * dist.clamp(min=0.0))


def _l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance, last dim is spatial coords."""
    return (a - b).norm(dim=-1)


def _lateral_dist(parcel_xy: torch.Tensor, bin_xy: torch.Tensor) -> torch.Tensor:
    return (parcel_xy - bin_xy).norm(dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Main reward function
# ──────────────────────────────────────────────────────────────────────────────

def example_dense_reward(env, obs, action: torch.Tensor, info: dict) -> torch.Tensor:
    """Staged wall-aware dense reward.  Called by env.compute_dense_reward().

    Parameters
    ----------
    env   : WarehouseSortEnv instance (gives access to geometry + actor state)
    obs   : observation dict (unused here; all state read from actors directly)
    action: (N, A) action tensor (unused)
    info  : dict returned by evaluate() — contains is_grasped, success_count, etc.

    Returns
    -------
    reward : (N,) float32 tensor
    """
    N = env.num_envs
    device = env.device
    reward = torch.zeros(N, device=device)

    # ── Geometry constants ──────────────────────────────────────────────────── #
    bx, by = env.bin_half          # bin footprint half-extents (0.11, 0.13)
    # bin_pos: (N, 2, 3) — [red_bin_xyz, blue_bin_xyz]
    bin_pos = env._bin_positions()   # uses env method, stays correct after bin-swap

    tcp_pos = env.agent.tcp_pose.p   # (N, 3)

    # ── Lazy-init episode bookkeeping ──────────────────────────────────────── #
    # We track per-parcel phase on the env object so bonuses fire only once.
    # Phase codes: 0=idle/done, 1=reach, 2=carry (to waypoint), 3=drop
    if not hasattr(env, '_rwd_phase') or env._rwd_phase.shape != (N, env.num_parcels):
        env._rwd_phase = torch.ones(N, env.num_parcels, dtype=torch.long, device=device)
        env._rwd_bonus_given = torch.zeros(N, env.num_parcels, 3, dtype=torch.bool, device=device)

    # ── Per-parcel loop ─────────────────────────────────────────────────────── #
    # Find the index of the first non-idle parcel in each env (the "current target").
    # We reward only that parcel's continuous potential, but any parcel's bonus fires
    # immediately when earned (so simultaneous lucky placements are still credited).

    for j, parcel in enumerate(env.parcels):
        p_pos = parcel.pose.p          # (N, 3)
        tag   = env.parcel_tags[:, j]  # (N,) colour id: 0=red, 1=blue

        # Correct bin xyz for each env (routes by tag id, handles bin-swap).
        correct_bin = bin_pos[torch.arange(N, device=device), tag]   # (N, 3)
        other_bin   = bin_pos[torch.arange(N, device=device), 1 - tag]

        # ── Settle check (same as evaluate, deterministic) ───────────────── #
        def _inside(bxyz):
            return (
                (torch.abs(p_pos[:, 0] - bxyz[:, 0]) < bx)
                & (torch.abs(p_pos[:, 1] - bxyz[:, 1]) < by)
                & (p_pos[:, 2] < SETTLED_Z)
                & (p_pos[:, 2] > 0.0)
            )

        in_correct = _inside(correct_bin)   # (N,) bool
        in_other   = _inside(other_bin)     # (N,) bool

        # ── Phase transitions ────────────────────────────────────────────── #
        # IDLE (0): parcel correctly placed — nothing to do.
        newly_idle = in_correct & (env._rwd_phase[:, j] != 0)
        env._rwd_phase[:, j] = torch.where(newly_idle, torch.zeros_like(env._rwd_phase[:, j]),
                                            env._rwd_phase[:, j])

        # Grasp state for this specific parcel.
        is_grasping_j = env.agent.is_grasping(parcel)   # (N,) bool

        # REACH → CARRY: parcel is grasped and we were in reach phase.
        go_carry = is_grasping_j & (env._rwd_phase[:, j] == 1)
        env._rwd_phase[:, j] = torch.where(go_carry,
                                            torch.full_like(env._rwd_phase[:, j], 2),
                                            env._rwd_phase[:, j])

        # CARRY → DROP: parcel is above the correct bin laterally AND above the rim.
        lateral = _lateral_dist(p_pos[:, :2], correct_bin[:, :2])
        above_rim = (p_pos[:, 2] >= WAYPOINT_Z) & (lateral < LATERAL_TOL)
        go_drop = is_grasping_j & (env._rwd_phase[:, j] == 2) & above_rim
        env._rwd_phase[:, j] = torch.where(go_drop,
                                            torch.full_like(env._rwd_phase[:, j], 3),
                                            env._rwd_phase[:, j])

        # If we lost the grasp while in CARRY or DROP, fall back to REACH.
        lost_grasp = (~is_grasping_j) & (env._rwd_phase[:, j] >= 2) & (~in_correct)
        env._rwd_phase[:, j] = torch.where(lost_grasp,
                                            torch.ones_like(env._rwd_phase[:, j]),
                                            env._rwd_phase[:, j])

        phase = env._rwd_phase[:, j]   # (N,) current phase

        # ── Continuous potentials ─────────────────────────────────────────── #

        # REACH: TCP toward parcel top face.
        parcel_top = p_pos.clone(); parcel_top[:, 2] += env.parcel_half[2]
        reach_pot = _tanh_pot(_l2(tcp_pos, parcel_top), K_REACH)

        # CARRY: parcel toward the clearance waypoint (directly above correct bin).
        waypoint = correct_bin.clone(); waypoint[:, 2] = WAYPOINT_Z
        carry_pot = _tanh_pot(_l2(p_pos, waypoint), K_CARRY)

        # DROP: parcel toward the bin floor center (once it's above the rim).
        drop_target = correct_bin.clone()   # z already at bin floor level
        drop_pot = _tanh_pot(_l2(p_pos[:, :2], drop_target[:, :2]), K_DROP) * \
                   _tanh_pot((p_pos[:, 2] - 0.02).clamp(min=0), K_DROP)

        is_reach  = (phase == 1).float()
        is_carry  = (phase == 2).float()
        is_drop   = (phase == 3).float()

        pot = POT_SCALE * (
            is_reach * reach_pot +
            is_carry * carry_pot +
            is_drop  * drop_pot
        )
        reward += pot

        # ── Bonuses (fire once, tracked by _rwd_bonus_given) ─────────────── #
        # Bonus 0: grasp
        b0 = go_carry & ~env._rwd_bonus_given[:, j, 0]
        reward += b0.float() * BONUS_GRASP
        env._rwd_bonus_given[:, j, 0] |= b0

        # Bonus 1: cleared the rim above correct bin
        b1 = go_drop & ~env._rwd_bonus_given[:, j, 1]
        reward += b1.float() * BONUS_ABOVE
        env._rwd_bonus_given[:, j, 1] |= b1

        # Bonus 2: correctly settled inside bin
        b2 = in_correct & ~env._rwd_bonus_given[:, j, 2]
        reward += b2.float() * BONUS_PLACE
        env._rwd_bonus_given[:, j, 2] |= b2

        # Penalty: mis-sorted (goes into wrong bin) — discourages random dropping.
        reward -= in_other.float() * (BONUS_PLACE * 0.5)

    # ── All-placed bonus ─────────────────────────────────────────────────────── #
    # One extra bonus when every parcel is correctly placed in this step.
    all_correct = info["success_count"] >= env.num_parcels
    if not hasattr(env, '_rwd_all_bonus_given'):
        env._rwd_all_bonus_given = torch.zeros(N, dtype=torch.bool, device=device)
    b_all = all_correct & ~env._rwd_all_bonus_given
    reward += b_all.float() * BONUS_PLACE
    env._rwd_all_bonus_given |= b_all

    return reward
