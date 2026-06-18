"""warehouse_sort/reward.py — staged, wall-aware dense reward for WarehouseSort-v1.

EXAMPLE DENSE REWARD (state-based, ManiSkill PickCube style). Simplest readable version,
not optimised. Replace/improve with your own — this is the core of your submission.

Design principle
----------------
Each parcel is a reach → grasp → carry-over-the-rim → **release-inside** sub-sequence. We shape
this like ManiSkill's ``PickCube`` reward — one **cumulative, monotonic** signal where every
step of progress *adds* reward and nothing has to be given back:

    reward = PLACED_CREDIT * (parcels correctly placed AND released)   # permanent, banked
           + stage_value(current target parcel)                        # in [0, PLACED_CREDIT]

The current target is the first parcel not yet *settled and released* in a bin. Its
``stage_value`` rises through the ordered sub-goals:

    reach (0→1) → grasp & carry to a clearance waypoint above the bin (1→2)
                → lower the held box toward the bin floor (2→HOLD_VALUE)
                → RELEASE so it settles inside (→ PLACED_CREDIT)

Lowering before release matters: it brings the box right down to the floor while still grasped,
so opening the gripper is a tiny, reliable drop that settles inside the bin — which makes the
release action easy for exploration to discover and low-risk to commit to, instead of dropping
from a height where the box can bounce back out.

Releasing is the final, highest-paying step: a parcel only counts (and only banks its credit)
once it is inside the correct bin **and no longer grasped**. This is deliberate — without it the
policy learns to grasp a box, lower it into the bin, and *keep holding* (the geometric "inside"
test is already satisfied), so it never frees the gripper for the next parcel and then drags the
first one back out when it moves. Tying the credit to release forces the open-gripper action and
frees the arm for the next parcel.

Why cumulative + ordered ceilings (not a phase-gated potential): a placed parcel's credit is
permanent and strictly larger than lingering in any earlier sub-goal, and each sub-goal's ceiling
is strictly higher than the previous one, so the policy can never do better by hovering. A
reward where every phase pays the same for being "in position" has a camping optimum and never
completes.

Wall-awareness: while grasped but not yet over the bin, the carry sub-goal targets a clearance
waypoint *above the bin rim* so the gradient lifts the parcel OVER the wall instead of wedging it
against the rim.

Hyperparameters are visible at the top. The reward is stateless (reads only the current scene),
so it is automatically correct under ManiSkill partial resets.
"""

from __future__ import annotations

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (all in SI / normalised units)
# ──────────────────────────────────────────────────────────────────────────────

K_REACH = 5.0    # sharpness of reach-to-parcel shaping
K_CARRY = 5.0    # sharpness of carry-to-waypoint shaping
K_LOWER = 5.0    # sharpness of lower-into-bin shaping (descent before release)

# Height of the clearance waypoint above the table surface (metres). Must clear the bin rim
# (full wall height = 2 * bin_wall_h = 0.05 m) with margin so the parcel goes OVER the wall.
WAYPOINT_Z = 0.12

# Lateral tolerance: the parcel must be within this radius of the bin centre to count as "over
# the bin" (ready to release), so it really is above the bin interior, not just high in the air.
LATERAL_TOL = 0.07   # metres; bin footprint half-extents are (0.11, 0.13)

# z threshold below which a parcel is considered "settled" inside a bin (matches evaluate()).
SETTLED_Z = 0.06

# Per-parcel credit, banked permanently once the parcel is correctly placed AND released. Equals
# the top of stage_value so the hand-off is seamless; the release step itself is the biggest
# single jump (HOLD_VALUE → PLACED_CREDIT).
PLACED_CREDIT = 3.0
HOLD_VALUE = 2.0     # stage value while grasped, lowered, and held at the bin floor (pre-release)
OVER_BIN_BASE = 1.5  # stage value the instant the box is over the bin (before lowering)

# Penalty for a parcel released in the WRONG-colour bin (discourages dumping).
MIS_PENALTY = 1.0


def example_dense_max(num_parcels: int) -> float:
    """Upper bound on reward (all parcels correctly placed and released). Normaliser to [0,1]."""
    return float(num_parcels) * PLACED_CREDIT


def _tanh_pot(dist: torch.Tensor, k: float) -> torch.Tensor:
    """Smooth potential ∈ (0, 1]: 1 when dist==0, decays with k."""
    return 1.0 - torch.tanh(k * dist.clamp(min=0.0))


def _l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).norm(dim=-1)


def example_dense_reward(env, obs, action: torch.Tensor, info: dict) -> torch.Tensor:
    """Cumulative, wall-aware, release-completing dense reward. Called by compute_dense_reward().

    Returns a (N,) float32 reward tensor.
    """
    N = env.num_envs
    device = env.device
    idx = torch.arange(N, device=device)

    bx, by = env.bin_half                 # bin footprint half-extents
    bin_pos = env._bin_positions()        # (N, 2, 3) — [red_bin_xyz, blue_bin_xyz], post-swap
    tcp_pos = env.agent.tcp_pose.p        # (N, 3)

    P = env.num_parcels
    grasped = torch.zeros(N, P, device=device, dtype=torch.bool)
    reach = torch.zeros(N, P, device=device)
    stage_g = torch.zeros(N, P, device=device)   # stage value while grasped, in [1, HOLD_VALUE]

    for j, parcel in enumerate(env.parcels):
        p_pos = parcel.pose.p                       # (N, 3)
        tag = env.parcel_tags[:, j]                 # (N,) colour id
        correct_bin = bin_pos[idx, tag]             # (N, 3)

        grasped[:, j] = env.agent.is_grasping(parcel)

        # reach: TCP toward the parcel's top face (live sub-goal before grasping).
        parcel_top = p_pos.clone()
        parcel_top[:, 2] += env.parcel_half[2]
        reach[:, j] = _tanh_pot(_l2(tcp_pos, parcel_top), K_REACH)

        # carry: parcel toward the clearance waypoint above the correct bin (over the rim).
        waypoint = correct_bin.clone()
        waypoint[:, 2] = WAYPOINT_Z
        carry = _tanh_pot(_l2(p_pos, waypoint), K_CARRY)

        # over the bin footprint (and roughly at the waypoint) → ready to lower + release.
        over_bin = _l2(p_pos[:, :2], correct_bin[:, :2]) < LATERAL_TOL
        # descent: drive the held box down to the bin floor (only counts once over the footprint).
        lower = _tanh_pot((p_pos[:, 2] - 0.04).clamp(min=0.0), K_LOWER)
        # ordered ceiling while grasped: carry climbs to 2 → over the bin we step DOWN to
        # OVER_BIN_BASE and climb back to HOLD_VALUE by lowering, so the box ends right at the
        # floor; the only way past HOLD_VALUE is to RELEASE (banked PLACED_CREDIT below).
        over_bin_stage = OVER_BIN_BASE + (HOLD_VALUE - OVER_BIN_BASE) * lower
        stage_g[:, j] = torch.where(over_bin, over_bin_stage, 1.0 + carry)

    # Placement is LATCHED by evaluate() (runs just before this): a parcel that has been
    # correctly placed AND released stays "done" for the rest of the episode. Reading the latch
    # (rather than the live geometry) means once a parcel is in its bin the arm gets ZERO further
    # reward for staying near it — the only remaining reward is reaching/placing the next parcel.
    placed_correct = env._placed_correct          # (N, P) latched
    placed_other = env._placed_other
    settled = placed_correct | placed_other

    # ── banked credit: permanent reward for every correctly placed+released parcel ─────────── #
    reward = PLACED_CREDIT * placed_correct.float().sum(dim=1)
    reward = reward - MIS_PENALTY * placed_other.float().sum(dim=1)

    # ── live shaping on the current target = first parcel not yet settled+released ─────────── #
    not_settled = ~settled
    any_target = not_settled.any(dim=1)
    target_j = torch.argmax(not_settled.float(), dim=1)
    g = grasped[idx, target_j]
    stage = torch.where(g, stage_g[idx, target_j], reach[idx, target_j])
    reward = reward + any_target.float() * stage

    return reward
