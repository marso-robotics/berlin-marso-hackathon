"""Warehouse Colour-Sort hackathon starter package.

Importing this package registers the ``WarehouseSort-v1`` ManiSkill environment.
"""

from warehouse_sort.env import WarehouseSortEnv  # noqa: F401  (registers WarehouseSort-v1)
from warehouse_sort.simple_sort_env import SimpleSortEnv  # noqa: F401  (registers SimpleSort-v1)

__all__ = ["WarehouseSortEnv", "SimpleSortEnv"]
