"""Warehouse Colour-Sort Pick-and-Place environment (ManiSkill 3).

A Franka Panda picks brown cardboard parcels from a central inbound zone and places each
into the output bin whose colour matches the parcel's *top-face tag* (red tag -> red bin,
blue tag -> blue bin). Score = number of parcels in the correct-colour bin.

Structure mirrors ManiSkill's built-in ``PickCube`` env (``_load_scene``,
``_initialize_episode``, ``evaluate``, the reward signature) and reuses its grasp-detection /
object-attachment pattern via ``self.agent.is_grasping(actor)`` (see PickCube.evaluate).

Difficulty is a pure config switch; every randomisation axis is an explicit, overridable
range (see BUILD_SPEC §8). This file contains NO tuned/shaping reward — only the sparse
default. The optional example dense reward lives in ``warehouse_sort/reward.py``.
"""

from typing import Any, Optional

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from warehouse_sort.reward import example_dense_reward, example_dense_max

# ---------------------------------------------------------------------------- #
# Colour identities. Index 0 = red, 1 = blue. These are the *base* RGB identities;
# appearance randomisation jitters the shade around them but keeps identity intact.
# ---------------------------------------------------------------------------- #
TAG_BASE_COLORS = [
    [0.80, 0.10, 0.10],  # 0: red
    [0.10, 0.20, 0.80],  # 1: blue
]
BIN_BASE_COLORS = [
    [0.85, 0.15, 0.15],  # 0: red bin
    [0.15, 0.25, 0.85],  # 1: blue bin
]
CARDBOARD_BASE = [0.62, 0.46, 0.30]  # warehouse brown


def _to_list(x):
    """OmegaConf lists / tensors / arrays -> plain python list."""
    if x is None:
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


@register_env("WarehouseSort-v1", max_episode_steps=100)
class WarehouseSortEnv(BaseEnv):
    """Colour-sort pick-and-place. One env class, parameterised by difficulty + ranges."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    SUPPORTED_REWARD_MODES = ["sparse", "dense", "normalized_dense", "none"]

    # ----- task geometry (metres) ----- #
    # Box ~0.052 m wide. This is "as big as the gripper allows" once yaw randomisation is
    # accounted for: the controller (pd_ee_delta_pos) can't rotate the gripper, so a square
    # box rotated by the max train yaw (~0.5 rad) presents an effective width of
    # 0.052*(cos+sin) ~= 0.071 m, still inside the ~0.08 m max opening. A 0.064 m box would
    # exceed the opening when rotated and become physically ungraspable.
    parcel_half = (0.026, 0.026, 0.03)   # half-extents of the cardboard box body
    tag_half = (0.010, 0.010, 0.0015)    # small coloured tag, placed on a top corner
    inbound_half = (0.10, 0.12)          # inbound zone half-extents (x, y) at table centre
    inbound_center = (0.0, 0.0)
    bin_half = (0.11, 0.13)              # bin floor footprint half-extents (x, y)
    bin_wall_h = 0.025                   # low walls (half-height; full height 0.05 m)
    bin_wall_t = 0.008
    bin_floor_t = 0.005
    bin_base_x = 0.0                     # bins sit at +/- bin_base_y on the y axis (robot's L/R)
    bin_base_y = 0.36                    # spaced clear of the central inbound zone (parcels)

    # Initial Panda arm pose. The gripper points straight DOWN (required for top-down grasping
    # with the fixed-orientation pd_ee_delta_pos controller); we only roll the final wrist joint
    # +pi/2 vs the ManiSkill default so the wrist camera's forward axis lines up with the table
    # (both parcels visible at the start). [joint1..7, finger1, finger2]; finger 0.04 = open.
    START_QPOS = [0.0, 0.3927, 0.0, -1.9635, 0.0, 2.356, 0.7854, 0.04, 0.04]

    def __init__(
        self,
        *args,
        difficulty: str = "easy",
        num_parcels: int = 2,
        fixed_poses: bool = True,
        randomization: Optional[dict] = None,
        reward_option: str = "sparse",     # "sparse" | "example_dense" (selects reward source)
        camera_width: int = 128,
        camera_height: int = 128,
        robot_init_qpos_noise: float = 0.02,
        robot_uids: str = "panda_wristcam",
        max_episode_steps: int = 100,
        **kwargs,
    ):
        # Stored for evaluate()'s speed metric. The gym TimeLimit (set via gym.make's
        # max_episode_steps, defaulting to the registered value) governs actual truncation.
        self.max_episode_steps = int(max_episode_steps)
        self.difficulty = difficulty
        self.num_parcels = int(num_parcels)
        self.fixed_poses = bool(fixed_poses)
        self.reward_option = reward_option
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._rand = self._normalize_rand(randomization)
        # tag assignment: balanced split across the 2 colours (>=1 of each).
        tags = [i % 2 for i in range(self.num_parcels)]
        self._parcel_tag_ids = tags
        # Override the Panda wrist camera ("hand_camera") resolution from config.
        sensor_configs = kwargs.pop("sensor_configs", {}) or {}
        sensor_configs = {
            **{"hand_camera": dict(width=self.camera_width, height=self.camera_height)},
            **sensor_configs,
        }
        super().__init__(*args, robot_uids=robot_uids, sensor_configs=sensor_configs, **kwargs)

    # ----------------------- randomisation config helpers ----------------------- #
    @staticmethod
    def _normalize_rand(r):
        """Fill any missing axis with a zero/empty (deterministic) default."""
        r = dict(r) if r else {}

        def g(d, k, default):
            d = r.get(d, {}) or {}
            return d.get(k, default)

        return dict(
            parcel_xy_jitter=_to_list(g("parcel_pose", "xy_jitter", [0.0, 0.0])),
            parcel_yaw_jitter=_to_list(g("parcel_pose", "yaw_jitter", [0.0, 0.0])),
            bin_side_swap_prob=float(g("bin_position", "side_swap_prob", 0.0)),
            bin_xy_jitter=_to_list(g("bin_position", "xy_jitter", [0.0, 0.0])),
            light_intensity=_to_list(g("lighting", "intensity", [1.0, 1.0])),
            light_dir_jitter=_to_list(g("lighting", "direction_jitter", [0.0, 0.0])),
            table_colors=_to_list(g("background", "table_colors", [[0.3, 0.3, 0.3]])),
            floor_colors=_to_list(g("background", "floor_colors", [[0.2, 0.2, 0.2]])),
            cardboard_shade=_to_list(g("appearance", "cardboard_shade", [0.0, 0.0])),
            tag_shade=_to_list(g("appearance", "tag_shade", [0.0, 0.0])),
            clutter_enabled=bool(g("clutter", "enabled", False)),
            clutter_contact_ok=bool(g("clutter", "contact_ok", False)),
        )

    # ----------------------------- cameras ------------------------------------- #
    @property
    def _default_sensor_configs(self):
        # Panda wrist camera ("hand_camera") is provided by the panda_wristcam robot.
        # A top-down sensor is added too so the scene is visually verifiable, but the
        # documented rgb obs uses the wrist camera (see _get_obs / README).
        return []

    @property
    def _default_human_render_camera_configs(self):
        # Third-person view of the workspace for human/video rendering.
        pose = sapien_utils.look_at(eye=[0.5, 0.0, 0.7], target=[0.0, 0.0, 0.05])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def render(self):
        """For render_mode='all', return a clean side-by-side [render camera | wrist camera]
        image (the policy's actual sensor view), instead of ManiSkill's padded tiling. Other
        render modes are unchanged."""
        if self.render_mode == "all":
            render = self.render_rgb_array()                       # (N, H, H, 3) uint8
            wrist = self.get_sensor_images()["hand_camera"]["rgb"]  # (N, h, w, 3) uint8
            h = render.shape[1]
            w = torch.nn.functional.interpolate(
                wrist.permute(0, 3, 1, 2).float(), size=(h, h), mode="nearest"
            ).permute(0, 2, 3, 1).to(render.dtype)
            return torch.cat([render, w], dim=2)                   # (N, H, 2H, 3)
        return super().render()

    # ----------------------------- agent --------------------------------------- #
    def _load_agent(self, options: dict):
        # Mount the Panda at the rear edge of the table, same offset PickCube uses.
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    # ----------------------------- lighting ------------------------------------ #
    def _load_lighting(self, options: dict):
        # Lighting is sampled once per reconfigure from the configured range (global to
        # the GPU scene). Intensity scales a fixed key/fill rig; direction is jittered.
        rng = self._batched_episode_rng
        lo, hi = self._rand["light_intensity"]
        inten = float(rng[0].uniform(lo, hi)) if hi > lo else float(lo)
        dlo, dhi = self._rand["light_dir_jitter"]
        dj = float(rng[0].uniform(dlo, dhi)) if dhi > dlo else 0.0
        self.scene.set_ambient_light([0.3 * inten, 0.3 * inten, 0.3 * inten])
        self.scene.add_directional_light(
            [1 + dj, 1 + dj, -1], [inten, inten, inten],
            shadow=self.enable_shadow, shadow_scale=5, shadow_map_size=2048,
        )
        self.scene.add_directional_light([0, 0, -1], [0.5 * inten] * 3)

    # ----------------------------- scene --------------------------------------- #
    def _build_bin(self, color, name: str):
        """A single low-walled bin actor (floor + 4 walls) built per-env so each scene
        can have its own appearance. Returns a merged batched Actor."""
        bx, by = self.bin_half
        h, t, ft = self.bin_wall_h, self.bin_wall_t, self.bin_floor_t
        builder = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=[*color, 1.0])
        # floor
        builder.add_box_collision(pose=sapien.Pose(p=[0, 0, ft]), half_size=[bx, by, ft])
        builder.add_box_visual(pose=sapien.Pose(p=[0, 0, ft]), half_size=[bx, by, ft], material=mat)
        # 4 walls
        walls = [
            ([bx, 0, h], [t, by, h]),
            ([-bx, 0, h], [t, by, h]),
            ([0, by, h], [bx, t, h]),
            ([0, -by, h], [bx, t, h]),
        ]
        for p, hs in walls:
            builder.add_box_collision(pose=sapien.Pose(p=p), half_size=hs)
            builder.add_box_visual(pose=sapien.Pose(p=p), half_size=hs, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        return builder.build_kinematic(name=name)

    def _build_parcel(self, idx: int, tag_id: int):
        """Brown cardboard box with a coloured tag on its top face. Built per parallel
        env so cardboard/tag shades can vary per scene (appearance randomisation).
        Tag colour identity == tag_id (0=red,1=blue); it is NOT the box colour."""
        phx, phy, phz = self.parcel_half
        thx, thy, thz = self.tag_half
        rng = self._batched_episode_rng
        per_env = []
        for i in range(self.num_envs):
            # per-scene shade jitter (keeps colour identity)
            cs_lo, cs_hi = self._rand["cardboard_shade"]
            ts_lo, ts_hi = self._rand["tag_shade"]
            c_off = float(rng[i].uniform(cs_lo, cs_hi)) if cs_hi > cs_lo else 0.0
            t_off = float(rng[i].uniform(ts_lo, ts_hi)) if ts_hi > ts_lo else 0.0
            cardboard = np.clip(np.array(CARDBOARD_BASE) + c_off, 0.05, 0.95)
            tagcol = np.clip(np.array(TAG_BASE_COLORS[tag_id]) + t_off, 0.05, 0.98)
            b = self.scene.create_actor_builder()
            b.add_box_collision(half_size=[phx, phy, phz])
            b.add_box_visual(
                half_size=[phx, phy, phz],
                material=sapien.render.RenderMaterial(base_color=[*cardboard.tolist(), 1.0]),
            )
            # small tag flush on a top CORNER of the top face (not covering the whole top)
            tag_x = phx - thx - 0.004
            tag_y = phy - thy - 0.004
            b.add_box_visual(
                pose=sapien.Pose(p=[-tag_x, tag_y, phz + thz]),
                half_size=[thx, thy, thz],
                material=sapien.render.RenderMaterial(base_color=[*tagcol.tolist(), 1.0]),
            )
            b.set_scene_idxs([i])
            b.initial_pose = sapien.Pose(p=[0, 0, phz + 0.5 * i])  # spread to avoid init overlap
            per_env.append(b.build_dynamic(name=f"parcel_{idx}_env{i}"))
        return Actor.merge(per_env, name=f"parcel_{idx}")

    def _build_background(self):
        """Recolour the table SURFACE (a full-size thin coloured top laid flush over the table
        top) and the floor, built per env so the palette varies per scene. The surface covers
        the whole table so it reads as the table's own colour, not a separate mat. Visual only;
        collision comes from the underlying ManiSkill table."""
        rng = self._batched_episode_rng
        table_colors = self._rand["table_colors"]
        floor_colors = self._rand["floor_colors"]
        tops, floors = [], []
        for i in range(self.num_envs):
            tc = table_colors[int(rng[i].randint(0, len(table_colors)))]
            fc = floor_colors[int(rng[i].randint(0, len(floor_colors)))]
            # full table top (ManiSkill table spans x~[-0.74,0.47], y~[-1.21,1.20]) laid flush;
            # y is the wide axis (~2.42 m), so the cover must extend ~1.25 m there.
            tb = self.scene.create_actor_builder()
            tb.add_box_visual(half_size=[0.66, 1.25, 0.0015],
                              material=sapien.render.RenderMaterial(base_color=[*tc, 1.0]))
            tb.set_scene_idxs([i])
            tb.initial_pose = sapien.Pose(p=[-0.135, 0, 0.0015])
            tops.append(tb.build_static(name=f"table_surface_env{i}"))
            fb = self.scene.create_actor_builder()
            fb.add_box_visual(half_size=[2.5, 2.5, 0.001],
                              material=sapien.render.RenderMaterial(base_color=[*fc, 1.0]))
            fb.set_scene_idxs([i])
            fb.initial_pose = sapien.Pose(p=[0, 0, -0.5])
            floors.append(fb.build_static(name=f"floor_mat_env{i}"))
        self.table_surface = Actor.merge(tops, name="table_surface")
        self.floor_mat = Actor.merge(floors, name="floor_mat")

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self._build_background()
        # two colour-coded bins, indexed by colour id (0=red, 1=blue)
        self.bins = [
            self._build_bin(BIN_BASE_COLORS[0], "bin_red"),
            self._build_bin(BIN_BASE_COLORS[1], "bin_blue"),
        ]
        # parcels, each with a fixed tag colour id
        self.parcels = [self._build_parcel(j, t) for j, t in enumerate(self._parcel_tag_ids)]
        # tag id per parcel as a (num_envs, num_parcels) tensor for evaluate()
        self.parcel_tags = torch.tensor(self._parcel_tag_ids, device=self.device).long()
        self.parcel_tags = self.parcel_tags[None].repeat(self.num_envs, 1)

    # --------------------------- episode init ---------------------------------- #
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # override the default arm pose: raised + wrist twisted so the wrist camera looks
            # forward over the inbound zone (both parcels visible from the start).
            qpos = torch.tensor(self.START_QPOS, device=self.device)
            self.agent.reset(qpos.unsqueeze(0).repeat(b, 1))
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # --- bins: per-env side assignment (+/- y) with optional swap + jitter --- #
            swap_p = self._rand["bin_side_swap_prob"]
            swap = torch.rand(b) < swap_p  # True -> red on -y side
            jx_lo, jx_hi = self._rand["bin_xy_jitter"]
            for color_id, bin_actor in enumerate(self.bins):
                # default: red(0) on +y (robot's left), blue(1) on -y. swap flips it.
                sign = torch.where(swap, torch.tensor(-1.0), torch.tensor(1.0))
                if color_id == 1:
                    sign = -sign
                pos = torch.zeros((b, 3))
                pos[:, 0] = self.bin_base_x
                pos[:, 1] = sign * self.bin_base_y
                if jx_hi > jx_lo:
                    pos[:, 0] += torch.rand(b) * (jx_hi - jx_lo) + jx_lo
                    pos[:, 1] += torch.rand(b) * (jx_hi - jx_lo) + jx_lo
                bin_actor.set_pose(Pose.create_from_pq(pos))

            # --- parcels: spawn in the inbound zone, tag kept top-visible --- #
            n = self.num_parcels
            cx, cy = self.inbound_center
            jx_lo, jx_hi = self._rand["parcel_xy_jitter"]
            yaw_lo, yaw_hi = self._rand["parcel_yaw_jitter"]
            # Deterministic grid centred on the inbound zone, with BOX-SIZE-AWARE spacing so
            # parcels never spawn overlapping (tags stay top-visible, no stacking) for any
            # parcel count. The grid stays COMPACT along x (depth from the robot base, the
            # costliest reach axis) and spreads along y (the robot's left/right sweep, which
            # has more reach headroom): at most 2 columns in x, remaining parcels along y.
            cols = min(2, n)
            rows = int(np.ceil(n / cols))
            # Spacing leaves room for the OPEN gripper (~0.08 m wide) to descend onto one
            # parcel without its fingers clipping a neighbour: gap >= gripper half-width +
            # margin (the 0.06 term), so adjacent boxes stay well clear during a top-down grasp.
            sx = 2 * self.parcel_half[0] + 0.06   # column spacing (m)
            sy = 2 * self.parcel_half[1] + 0.06   # row spacing (m)
            for j, parcel in enumerate(self.parcels):
                r, c = divmod(j, cols)
                gx = cx + (c - (cols - 1) / 2.0) * sx
                gy = cy + (r - (rows - 1) / 2.0) * sy
                pos = torch.zeros((b, 3))
                pos[:, 0] = gx
                pos[:, 1] = gy
                pos[:, 2] = self.parcel_half[2] + 0.001
                if not self.fixed_poses and jx_hi > jx_lo:
                    pos[:, 0] += torch.rand(b) * (jx_hi - jx_lo) + jx_lo
                    pos[:, 1] += torch.rand(b) * (jx_hi - jx_lo) + jx_lo
                # yaw about z only -> tag stays on top (top-visible). lock x,y rotation.
                if not self.fixed_poses and yaw_hi > yaw_lo:
                    yaw = torch.rand(b) * (yaw_hi - yaw_lo) + yaw_lo
                else:
                    yaw = torch.zeros(b)
                quat = torch.zeros((b, 4))
                quat[:, 0] = torch.cos(yaw / 2)
                quat[:, 3] = torch.sin(yaw / 2)
                parcel.set_pose(Pose.create_from_pq(pos, quat))

            # episode bookkeeping for sparse reward + speed metric
            self._prev_sorted = torch.zeros(self.num_envs, device=self.device)
            self._steps_to_complete = torch.full(
                (self.num_envs,), self.max_episode_steps, dtype=torch.long, device=self.device
            )

    # ------------------------------ observations ------------------------------- #
    def _bin_positions(self):
        """(num_envs, 2, 3): current xyz of bin colour 0 then colour 1."""
        return torch.stack([b.pose.p for b in self.bins], dim=1)

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            # privileged low-dim state (BUILD_SPEC §4). Ordering is documented in README.
            parcel_pose = torch.stack([p.pose.raw_pose for p in self.parcels], dim=1)  # (N, P, 7)
            tag_onehot = torch.nn.functional.one_hot(self.parcel_tags, num_classes=2).float()
            bin_pos = self._bin_positions()  # (N, 2, 3)
            bin_color_onehot = torch.eye(2, device=self.device)[None].repeat(self.num_envs, 1, 1)
            obs.update(
                parcel_pose=parcel_pose.reshape(self.num_envs, -1),
                parcel_tag=tag_onehot.reshape(self.num_envs, -1),
                bin_position=bin_pos.reshape(self.num_envs, -1),
                bin_color=bin_color_onehot.reshape(self.num_envs, -1),
            )
        return obs

    # ------------------------------ evaluate ----------------------------------- #
    def evaluate(self):
        # grasp detection reused from PickCube (per-parcel)
        is_grasped = torch.stack(
            [self.agent.is_grasping(p) for p in self.parcels], dim=1
        ).any(dim=1)

        bin_pos = self._bin_positions()  # (N, 2, 3)
        bx, by = self.bin_half
        # parcel must be settled low inside the bin footprint (not carried high above it)
        rim_z = 0.06

        correct = torch.zeros(self.num_envs, device=self.device)
        mis = torch.zeros(self.num_envs, device=self.device)
        placed = torch.zeros(self.num_envs, device=self.device)
        for j, parcel in enumerate(self.parcels):
            p = parcel.pose.p  # (N, 3)
            tag = self.parcel_tags[:, j]  # (N,)
            correct_bin = bin_pos[torch.arange(self.num_envs), tag]   # (N,3) matching colour
            other_bin = bin_pos[torch.arange(self.num_envs), 1 - tag]

            def inside(b):
                return (
                    (torch.abs(p[:, 0] - b[:, 0]) < bx)
                    & (torch.abs(p[:, 1] - b[:, 1]) < by)
                    & (p[:, 2] < rim_z)
                    & (p[:, 2] > 0.0)
                )

            in_correct = inside(correct_bin)
            in_other = inside(other_bin)
            correct += in_correct.float()
            mis += in_other.float()
            placed += (in_correct | in_other).float()

        all_placed = placed >= self.num_parcels
        # speed: first step at which all parcels are placed (per env)
        newly_done = all_placed & (self._steps_to_complete == self.max_episode_steps)
        self._steps_to_complete = torch.where(
            newly_done, self.elapsed_steps.long(), self._steps_to_complete
        )

        return {
            "success_count": correct,          # primary (float count per env)
            "mis_sort_count": mis,             # diagnostic
            "all_placed": all_placed,          # bool per env
            "steps_to_complete": self._steps_to_complete,
            "is_grasped": is_grasped,
            "success": all_placed,             # ManiSkill convention: episode "success"
        }

    # ------------------------------ rewards ------------------------------------ #
    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Default reward: +1 each time a parcel newly lands in its correct-colour bin."""
        cur = info["success_count"]
        delta = torch.clamp(cur - self._prev_sorted, min=0.0)
        self._prev_sorted = cur
        return delta

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # EXAMPLE DENSE REWARD lives in reward.py; the env just delegates to it.
        return example_dense_reward(self, obs, action, info)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / example_dense_max(self.num_parcels)
