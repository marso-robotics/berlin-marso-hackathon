"""Thin launcher for ManiSkill's replay_trajectory that first registers WarehouseSort-v1.

The stock ``python -m mani_skill.trajectory.replay_trajectory`` does a ``gym.make`` of the env
id stored in the demo's json, but it never imports our package, so the custom env is not
registered. This wrapper imports ``warehouse_sort`` (which registers the env) and then defers
entirely to ManiSkill's own replay CLI — we add registration, not logic.

Usage (identical flags to the real tool):
  pixi run python il/_replay.py --traj-path <demos>.h5 --obs-mode rgb --sim-backend physx_cuda --save-traj
"""

import warehouse_sort  # noqa: F401  (registers WarehouseSort-v1)
from mani_skill.trajectory.replay_trajectory import main, parse_args

if __name__ == "__main__":
    main(parse_args())
