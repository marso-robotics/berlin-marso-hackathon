"""Warehouse Colour-Sort hackathon starter package.

Importing this package registers the ``WarehouseSort-v1`` ManiSkill environment.
"""

from warehouse_sort.env import WarehouseSortEnv  # noqa: F401  (registers the env)

__all__ = ["WarehouseSortEnv"]
