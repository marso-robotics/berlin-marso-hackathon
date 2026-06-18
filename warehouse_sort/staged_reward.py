"""warehouse_sort/reward.py — simple staged dense reward for WarehouseSort-v1.

Designed like a scripted policy: at any instant exactly ONE parcel is "active"
(the first one not yet correctly placed), and the reward walks it up a fixed
ladder of stages.  When a parcel is finished the active pointer advances, so all
reward tied to the finished bin vanishes and shaping moves to the next parcel.

This file is STATELESS — reward is a pure function of the current sim state.
That is what keeps it simple: no per-episode bonus buffers, nothing to reset,
and (because each rung is a *level*, not a one-off delta) nothing to "farm".

Stage ladder for the active parcel  (maps to the 5 requested steps)
-------------------------------------------------------------------
  step 1  "move cube inside matching box":
      rung 0  REACH      tcp -> parcel
      rung 1  GRASP      +1 once grasped
      rung 2  OVER       parcel -> waypoint above the *correct* bin (up & over wall)
      rung 3  DESCEND    parcel -> bin floor, once it's above the rim & aligned
      rung 4  INSIDE     +1 once parcel is settled-low inside the correct bin
  step 2 / step 5  "encourage releasing -> placement means NOT grasped":
      rung 5  RELEASE    reward opening the gripper while inside; "done" = inside & ungrasped
  step 3  "no more reward for being near the place box":
      handled automatically — a done parcel is never the active one, so none of
      its terms are ever evaluated again.
  step 4  "move to the second box":
      also automatic — the active pointer advances to parcel 1, and its REACH
      term pulls the arm back to the inbound zone.

A floor of STAGE_DONE per finished parcel locks in progress so completed parcels
keep paying out and the agent is never tempted to undo them.
"""

from __future__ import annotations

import torch

# -- tunables ---------------------------------------------------------------- #
TANH_K       = 5.0    # sharpness of all distance potentials
WAYPOINT_Z   = 0.12   # clearance height above the table to clear the bin rim (0.05 m wall)
REST_Z       = 0.04   # parcel-center height when resting on the bin floor
LATERAL_TOL  = 0.07   # how close (xy) to bin center counts as "over the bin"
SETTLED_Z    = 0.06   # parcel-center z below which it counts as inside (matches evaluate())
STAGE_DONE   = 7.0    # reward floor added per finished parcel (must exceed max active rungs=6)
FINGER_OPEN  = 0.08   # sum of the two Panda finger joints when fully open


def example_dense_max(num_parcels: int) -> float:
    """Upper bound on per-step reward -> used to normalise to ~[0,1]."""
    return STAGE_DONE * float(num_parcels)


def _pot(dist: torch.Tensor) -> torch.Tensor:
    """Smooth potential in (0,1]: 1 at dist=0, decaying with TANH_K."""
    return 1.0 - torch.tanh(TANH_K * dist.clamp(min=0.0))


def example_dense_reward(env, obs, action, info: dict) -> torch.Tensor:
    N = env.num_envs
    dev = env.device
    idx = torch.arange(N, device=dev)

    bx, by = env.bin_half
    bin_pos = env._bin_positions()          # (N, 2, 3), routes correctly after bin-swap
    tcp = env.agent.tcp_pose.p              # (N, 3)

    # gripper openness in [0,1]: 1 = fully open, 0 = closed (last two qpos are the fingers)
    finger = env.agent.robot.get_qpos()[:, -2:].sum(dim=1)
    openness = (finger / FINGER_OPEN).clamp(0.0, 1.0)

    P = env.num_parcels

    # -- 1. classify every parcel: where is it, is it grasped, is it done? ----- #
    in_correct = torch.zeros(N, P, dtype=torch.bool, device=dev)
    grasped    = torch.zeros(N, P, dtype=torch.bool, device=dev)
    p_pos_all  = torch.zeros(N, P, 3, device=dev)
    cbin_all   = torch.zeros(N, P, 3, device=dev)

    for j, parcel in enumerate(env.parcels):
        p = parcel.pose.p                          # (N,3)
        tag = env.parcel_tags[:, j]                # (N,)
        cbin = bin_pos[idx, tag]                   # (N,3) correct-colour bin
        inside = (
            (torch.abs(p[:, 0] - cbin[:, 0]) < bx)
            & (torch.abs(p[:, 1] - cbin[:, 1]) < by)
            & (p[:, 2] < SETTLED_Z)
            & (p[:, 2] > 0.0)
        )
        p_pos_all[:, j] = p
        cbin_all[:, j] = cbin
        in_correct[:, j] = inside
        grasped[:, j] = env.agent.is_grasping(parcel)

    # placement is LATCHED by evaluate() (runs just before this each step): a parcel correctly
    # placed AND released stays "done" for the whole episode, even if the gripper later brushes
    # it. Reading the latch (not the live `in_correct & ~grasped`) is what guarantees a finished
    # parcel never becomes active again — so the arm gets zero further reward near its bin and is
    # pulled on to the next parcel, and a momentary grasp flicker can't un-place it / drag it out.
    placed_correct = env._placed_correct           # (N,P) latched (inside AND released)
    placed_other = env._placed_other               # (N,P) latched wrong-bin
    settled = placed_correct | placed_other        # finished (correct or mis-sorted)
    num_done = placed_correct.sum(dim=1).float()   # (N,)
    all_done = settled.all(dim=1)                  # (N,)

    # active parcel = first not-yet-settled (scripted order: parcel 0, then 1, ...)
    active_idx = torch.argmax((~settled).long(), dim=1)   # first True; 0 if none

    # -- 2. floor: lock in finished parcels ----------------------------------- #
    reward = STAGE_DONE * num_done

    # -- 3. shape ONLY the active parcel -------------------------------------- #
    for j in range(P):
        active = (active_idx == j) & ~all_done     # (N,) which envs have parcel j active
        if not active.any():
            continue

        p = p_pos_all[:, j]
        cbin = cbin_all[:, j]
        g = grasped[:, j]
        ins = in_correct[:, j]

        # rung 0 — REACH: tcp toward the parcel's top face
        parcel_top = p.clone(); parcel_top[:, 2] += env.parcel_half[2]
        rung = _pot((tcp - parcel_top).norm(dim=-1))          # [0,1]

        # rung 1 — GRASP
        rung = rung + g.float()                                # +1

        # rung 2 — OVER: while grasped, lift parcel to waypoint above the correct bin
        waypoint = cbin.clone(); waypoint[:, 2] = WAYPOINT_Z
        over_pot = _pot((p - waypoint).norm(dim=-1))           # [0,1]
        rung = rung + g.float() * over_pot

        # rung 3 — DESCEND: once aligned over the bin footprint, lower toward the floor. The gate
        # is ONLY lateral alignment (not a height window): descend_pot rewards smaller z, so the
        # reward must keep paying out all the way DOWN to the floor — gating it to "still high"
        # would switch the reward off mid-descent and strand the box at rim height (never inside).
        lateral = (p[:, :2] - cbin[:, :2]).norm(dim=-1)
        aligned = lateral < LATERAL_TOL
        descend_pot = _pot((p[:, 2] - REST_Z).clamp(min=0.0))  # [0,1], 1 at the bin floor
        rung = rung + (g & aligned).float() * descend_pot

        # rung 4 — INSIDE: parcel settled low inside the correct bin
        rung = rung + ins.float()                              # +1

        # rung 5 — RELEASE (step 2/5): while inside & still grasped, reward opening
        rung = rung + (ins & g).float() * openness             # [0,1]

        reward = reward + active.float() * rung

    # penalise a parcel released into the WRONG-colour bin (latched, discourages guessing)
    reward = reward - placed_other.float().sum(dim=1) * (STAGE_DONE * 0.5)

    return reward