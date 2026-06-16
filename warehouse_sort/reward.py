# EXAMPLE DENSE REWARD (state-based, ManiSkill StackCube/PickCube style). Simplest readable
# version, not optimised. Replace/improve with your own -- this is the core of your submission.
#
# Mirrors the staged structure of ManiSkill's StackCube reward, where each stage overwrites the
# reward with a strictly higher base so later stages dominate earlier ones, for the parcel
# currently being worked (the nearest unsorted parcel):
#   stage 1 reach   : base up to 2          (1 - tanh of TCP->parcel distance)
#   stage 2 grasp   : base 4 + place        (grasped -> carry toward a goal above the bin)
#   stage 3 in-bin  : base 6 + ungrasp      (parcel settled in bin -> reward OPENING the gripper
#                                            so the parcel is released and stays sorted)
# plus a large bonus (8) per parcel already correctly sorted, so the policy does all of them.
# Like StackCube, the ungrasp term is what teaches release. NO vision; not tuned for efficiency.

import torch

_OVER_Z = 0.14   # carry-goal height when approaching from outside the bin (clears the wall)
_DROP_Z = 0.04   # carry-goal height once over the footprint (settle the parcel into the bin)
_FINGER_OPEN = 0.04   # panda finger joint open value (per finger)


def example_dense_max(num_parcels: int) -> float:
    """Max value of example_dense_reward for one env (used to normalise to ~[0,1]):
    9 per sorted parcel, + up to ~9 for the parcel currently being worked."""
    return 9.0 * num_parcels + 9.0


def example_dense_reward(env, obs, action, info):
    device = env.device
    n = env.num_envs
    ar = torch.arange(n, device=device)
    tcp = env.agent.tcp_pose.p                                          # (n, 3)
    bin_pos = env._bin_positions()                                      # (n, 2, 3)

    bx, by = env.bin_half
    rim_z = 0.06
    parcel_p = torch.stack([p.pose.p for p in env.parcels], dim=1)      # (n, P, 3)
    tags = env.parcel_tags                                              # (n, P)
    correct_bin = bin_pos[ar[:, None], tags]                           # (n, P, 3)

    # parcels already correctly sorted (settled inside their colour bin)
    sorted_mask = (
        (torch.abs(parcel_p[..., 0] - correct_bin[..., 0]) < bx)
        & (torch.abs(parcel_p[..., 1] - correct_bin[..., 1]) < by)
        & (parcel_p[..., 2] < rim_z)
        & (parcel_p[..., 2] > 0.0)
    )
    num_sorted = sorted_mask.float().sum(dim=1)                         # (n,)
    all_sorted = sorted_mask.all(dim=1)

    # active parcel = nearest unsorted parcel to the TCP
    tcp_to_parcel = torch.linalg.norm(parcel_p - tcp[:, None, :], dim=-1)
    tcp_to_parcel = tcp_to_parcel + sorted_mask.float() * 1e3
    active = torch.argmin(tcp_to_parcel, dim=1)                        # (n,)
    active_p = parcel_p[ar, active]                                     # (n, 3)
    active_bin = correct_bin[ar, active]                               # (n, 3)
    active_in_bin = sorted_mask[ar, active]
    is_grasped = torch.stack(
        [env.agent.is_grasping(p) for p in env.parcels], dim=1
    )[ar, active]

    # stage 1: reach the active parcel  (base up to 2)
    reach_dist = torch.linalg.norm(active_p - tcp, dim=-1)
    reward = 2 * (1 - torch.tanh(5 * reach_dist))
    # ...while keeping the gripper OPEN as it approaches, so it can descend onto the box and
    # enclose it (avoids the "hover with a closed gripper" trap). Small term, dominated by grasp.
    fingers = env.agent.robot.get_qpos()[:, -2:]
    open_frac = (fingers.sum(dim=1) / (2 * _FINGER_OPEN)).clamp(0, 1)
    near = reach_dist < 0.08
    reward = reward + 0.5 * open_frac * near.float()

    # stage 2: grasped -> move the parcel to its bin in explicit LIFT -> CARRY -> DROP sub-stages,
    # each with a higher base so the policy is rewarded for lifting BEFORE moving laterally (a
    # single 3D goal lets it just drag the box along the table into the wall). bx/by give the
    # footprint; the parcel must clear the low wall (~0.05 m) so "lifted" ~ box centre above 0.10.
    box_z = active_p[:, 2]
    xy_dist = torch.linalg.norm(active_p[:, :2] - active_bin[:, :2], dim=-1)
    over_footprint = (torch.abs(active_p[:, 0] - active_bin[:, 0]) < bx) & \
                     (torch.abs(active_p[:, 1] - active_bin[:, 1]) < by)

    # lift + carry, CONTINUOUS (no hard threshold to get trapped at): reward height, and credit
    # lateral progress toward the bin scaled by how lifted the box is -- so dragging the box along
    # the table earns little, but lifting then moving over earns the full carry. base 4..6.
    lift_term = (box_z / _OVER_Z).clamp(0, 1)
    grasp_reward = 4 + lift_term + lift_term * (1 - torch.tanh(2.0 * xy_dist))
    # once over the footprint, switch to lowering the box into the bin. base 6..7.
    drop_prog = 1 - ((box_z - _DROP_Z).clamp(min=0) / _OVER_Z).clamp(0, 1)
    grasp_reward = torch.where(over_footprint, 6 + drop_prog, grasp_reward)
    reward = torch.where(is_grasped, grasp_reward, reward)

    # stage 3: parcel settled in the bin -> reward OPENING the gripper (release), base 8
    ungrasp = torch.where(is_grasped, open_frac, torch.ones_like(open_frac))
    reward = torch.where(active_in_bin, 8 + ungrasp, reward)

    # accumulate a large bonus per parcel already correctly sorted; clamp at the max
    reward = reward + 9.0 * num_sorted
    reward = torch.where(all_sorted, torch.full_like(reward, example_dense_max(env.num_parcels)), reward)
    return reward
