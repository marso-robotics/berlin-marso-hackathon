"""Simple single-parcel colour-sort pick-and-place (ManiSkill 3) — a SHORT-HORIZON
ablation of WarehouseSort-v1.

Motivation
----------
WarehouseSort-v1 sorts 2-8 parcels per episode. That makes it a long-horizon, multi-stage
task (pick / place / repeat), which is hard to learn from scratch. This file strips the task
down to its atomic unit to test whether a simpler task ladder trains more reliably:

    * exactly ONE parcel in the inbound zone (a brown cardboard box with a RED tag),
    * TWO possible destinations — a red bin and a blue bin (left/right of the robot),
    * the task: pick the red parcel and place it in the RED box (the colour-matched one),
    * then drop it and return the arm to its starting pose.

It mirrors ManiSkill's built-in ``PickCube`` in structure (``_load_scene``,
``_initialize_episode``, ``evaluate``, the dense-reward signature) and reuses its grasp
detection (``self.agent.is_grasping``). Everything — env AND reward — lives in this one file,
the way the stock ManiSkill tasks are laid out. The dense reward is a simple PickCube-style
staged ladder (reach -> grasp -> place -> drop -> return home -> static), NOT tuned for sample
efficiency — it's the readable baseline we want to compare task ladders with.

Train it with the SAME harness as WarehouseSort-v1 — no new trainer:

    python train.py difficulty=simple reward=example_dense
"""

from typing import Any

import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

# ---------------------------------------------------------------------------- #
# Colour identities (match WarehouseSort-v1). The parcel always carries the RED tag,
# so the colour-matched destination is always the red bin.
# ---------------------------------------------------------------------------- #
RED = [0.85, 0.15, 0.15]
BLUE = [0.15, 0.25, 0.85]
CARDBOARD = [0.62, 0.46, 0.30]  # warehouse brown


@register_env("SimpleSort-v1", max_episode_steps=80)
class SimpleSortEnv(BaseEnv):
    """One red parcel, two bins (red + blue). Pick the parcel, place it in the red bin,
    release, and return the arm home. Single registered env, PickCube-style."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    SUPPORTED_REWARD_MODES = ["sparse", "dense", "normalized_dense", "none"]

    # ----- task geometry (metres), identical to WarehouseSort-v1 so results compare ----- #
    parcel_half = (0.026, 0.026, 0.03)   # half-extents of the cardboard box body
    tag_half = (0.010, 0.010, 0.0015)    # small coloured tag on a top corner
    inbound_center = (0.0, 0.0)          # parcel starts at table centre
    bin_half = (0.11, 0.13)              # bin floor footprint half-extents (x, y)
    bin_wall_h = 0.025                   # low walls (half-height; full height 0.05 m)
    bin_wall_t = 0.008
    bin_floor_t = 0.005
    bin_base_x = 0.0
    bin_base_y = 0.22                    # red bin at +y, blue bin at -y (close to the parcel so
                                         # the place + return-home sequence is short to learn)

    # geometric placement check (mirrors WarehouseSort-v1.evaluate)
    rim_z = 0.06                         # parcel-centre z below which it counts as "in the bin"
    rest_z = 0.04                        # parcel-centre z resting on the bin floor (place target)

    # Initial Panda arm pose: gripper points straight DOWN (top-down grasp for the
    # fixed-orientation pd_ee_delta_pos controller). [joint1..7, finger1, finger2]; 0.04 = open.
    START_QPOS = [0.0, 0.3927, 0.0, -1.9635, 0.0, 2.356, 0.7854, 0.04, 0.04]

    def __init__(self, *args, robot_init_qpos_noise=0.02, robot_uids="panda", **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ----------------------------- cameras ------------------------------------- #
    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0.0, 0.7], target=[0.0, 0.0, 0.05])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    # ----------------------------- agent --------------------------------------- #
    def _load_agent(self, options: dict):
        # Mount the Panda at the rear edge of the table (same offset PickCube uses).
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    # ----------------------------- scene --------------------------------------- #
    def _build_bin(self, color, name: str):
        """A single low-walled bin (floor + 4 walls), kinematic."""
        bx, by = self.bin_half
        h, t, ft = self.bin_wall_h, self.bin_wall_t, self.bin_floor_t
        b = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=[*color, 1.0])
        b.add_box_collision(pose=sapien.Pose(p=[0, 0, ft]), half_size=[bx, by, ft])
        b.add_box_visual(pose=sapien.Pose(p=[0, 0, ft]), half_size=[bx, by, ft], material=mat)
        for p, hs in [
            ([bx, 0, h], [t, by, h]), ([-bx, 0, h], [t, by, h]),
            ([0, by, h], [bx, t, h]), ([0, -by, h], [bx, t, h]),
        ]:
            b.add_box_collision(pose=sapien.Pose(p=p), half_size=hs)
            b.add_box_visual(pose=sapien.Pose(p=p), half_size=hs, material=mat)
        b.initial_pose = sapien.Pose(p=[0, 0, 0])
        return b.build_kinematic(name=name)

    def _build_parcel(self):
        """Brown cardboard box with a small RED tag on a top corner."""
        phx, phy, phz = self.parcel_half
        thx, thy, thz = self.tag_half
        b = self.scene.create_actor_builder()
        b.add_box_collision(half_size=[phx, phy, phz])
        b.add_box_visual(half_size=[phx, phy, phz],
                         material=sapien.render.RenderMaterial(base_color=[*CARDBOARD, 1.0]))
        tag_x = phx - thx - 0.004
        tag_y = phy - thy - 0.004
        b.add_box_visual(pose=sapien.Pose(p=[-tag_x, tag_y, phz + thz]),
                         half_size=[thx, thy, thz],
                         material=sapien.render.RenderMaterial(base_color=[*RED, 1.0]))
        b.initial_pose = sapien.Pose(p=[0, 0, phz + 0.001])
        return b.build_dynamic(name="parcel")

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.red_bin = self._build_bin(RED, "bin_red")
        self.blue_bin = self._build_bin(BLUE, "bin_blue")
        self.parcel = self._build_parcel()
        # arm home pose (first 7 joints) used by the "return home" reward stage
        self._home_qpos = torch.tensor(self.START_QPOS[:7], device=self.device)

    # --------------------------- episode init ---------------------------------- #
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos = torch.tensor(self.START_QPOS, device=self.device)
            self.agent.reset(qpos.unsqueeze(0).repeat(b, 1))
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # bins: red at +y, blue at -y (fixed — this is the EASY ablation)
            red_p = torch.zeros((b, 3)); red_p[:, 0] = self.bin_base_x; red_p[:, 1] = self.bin_base_y
            blue_p = torch.zeros((b, 3)); blue_p[:, 0] = self.bin_base_x; blue_p[:, 1] = -self.bin_base_y
            self.red_bin.set_pose(Pose.create_from_pq(red_p))
            self.blue_bin.set_pose(Pose.create_from_pq(blue_p))

            # parcel: fixed pose at the inbound centre, flat on the table
            cx, cy = self.inbound_center
            pos = torch.zeros((b, 3))
            pos[:, 0] = cx; pos[:, 1] = cy; pos[:, 2] = self.parcel_half[2] + 0.001
            self.parcel.set_pose(Pose.create_from_pq(pos))

    # ------------------------- goal / helpers ---------------------------------- #
    def _red_bin_goal(self):
        """(N,3): point at the red bin centre at resting height — the place target."""
        g = self.red_bin.pose.p.clone()
        g[:, 2] = self.rest_z
        return g

    def _in_red_bin(self):
        """(N,) bool: parcel body settled low inside the red bin footprint."""
        p = self.parcel.pose.p
        c = self.red_bin.pose.p
        bx, by = self.bin_half
        return (
            (torch.abs(p[:, 0] - c[:, 0]) < bx)
            & (torch.abs(p[:, 1] - c[:, 1]) < by)
            & (p[:, 2] < self.rim_z)
            & (p[:, 2] > 0.0)
        )

    def _home_dist(self):
        """(N,) L2 distance of the 7 arm joints from the start pose."""
        return (self.agent.robot.get_qpos()[:, :7] - self._home_qpos).norm(dim=1)

    # ------------------------------ observations ------------------------------- #
    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if "state" in self.obs_mode:
            goal = self._red_bin_goal()
            obs.update(
                parcel_pose=self.parcel.pose.raw_pose,
                tcp_to_parcel=self.parcel.pose.p - self.agent.tcp_pose.p,
                parcel_to_goal=goal - self.parcel.pose.p,
                red_bin_pos=self.red_bin.pose.p,
                blue_bin_pos=self.blue_bin.pose.p,
                home_qpos_err=(self.agent.robot.get_qpos()[:, :7] - self._home_qpos),
            )
        return obs

    # ------------------------------ evaluate ----------------------------------- #
    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.parcel)
        in_bin = self._in_red_bin()
        is_placed = in_bin & ~is_grasped            # dropped in the correct box
        is_home = self._home_dist() < 0.3           # arm back near its start pose
        is_static = self.agent.is_static(0.2)
        # full task: parcel placed in the red bin, released, and the arm returned home & settled
        success = is_placed & is_home & is_static
        return {
            "success": success,
            "success_count": is_placed.float(),   # task score: parcel in correct bin (placed)
            "in_bin": in_bin,
            "is_placed": is_placed,
            "is_home": is_home,
            "is_static": is_static,
            "is_grasped": is_grasped,
        }

    # ------------------------------ rewards ------------------------------------ #
    # PickCube-style staged dense reward, written inline (this is the reference baseline).
    # The ladder is built so total reward STRICTLY INCREASES along the whole task — including
    # AFTER the parcel is placed, so the agent is pulled through the endgame the prior version
    # got stuck before: release the gripper, then return the arm home, then settle.
    #
    #   reach -> [+1 hold] -> place -> [+1 in-bin] -> release(open) -> [+1 dropped] -> home -> static
    #
    # The three "+1" milestones (hold / in-bin / dropped) lock in each subgoal so a later stage
    # never costs reward (e.g. opening the gripper to drop keeps the hold credit via the in-bin
    # milestone), which is what lets the policy commit to letting go instead of hovering.
    MAX_REWARD = 10.0

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp = self.agent.tcp_pose.p
        parcel = self.parcel.pose.p
        is_grasped = info["is_grasped"]
        in_bin = info["in_bin"]
        dropped = info["is_placed"]                  # in_bin & ~grasped (parcel released in bin)
        held_or_in = (is_grasped | in_bin).float()

        # 1) REACH — tcp toward the parcel's top face
        parcel_top = parcel.clone(); parcel_top[:, 2] += self.parcel_half[2]
        reward = 1 - torch.tanh(5 * (parcel_top - tcp).norm(dim=1))

        # 2) HOLD milestone (+1) — grasped the parcel; kept once it's in the bin.
        reward = reward + held_or_in

        # 3) PLACE — parcel toward the red-bin goal; kept while grasped OR already in the bin.
        place = 1 - torch.tanh(5 * (self._red_bin_goal() - parcel).norm(dim=1))
        reward = reward + place * held_or_in

        # 4) IN-BIN milestone (+1) — parcel is geometrically inside the red bin.
        reward = reward + in_bin.float()

        # 5) RELEASE — once in the bin, reward opening the gripper (last 2 qpos = fingers).
        openness = (self.agent.robot.get_qpos()[:, -2:].sum(dim=1) / 0.08).clamp(0.0, 1.0)
        reward = reward + openness * in_bin.float()

        # 6) DROPPED milestone (+1) — parcel released inside the bin (gripper let go).
        reward = reward + dropped.float()

        # 7) RETURN HOME — after dropping, reward bringing the arm back to its start pose.
        home = 1 - torch.tanh(self._home_dist())
        reward = reward + home * dropped.float()

        # 8) STATIC — after dropping, reward settling (low joint velocity).
        qvel = self.agent.robot.get_qvel()[:, :-2]
        static = 1 - torch.tanh(5 * qvel.norm(dim=1))
        reward = reward + static * dropped.float()

        # full task done (placed + released + home + settled) -> peak reward, episode terminates.
        reward[info["success"]] = self.MAX_REWARD
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / self.MAX_REWARD
