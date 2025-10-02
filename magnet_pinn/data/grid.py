"""
NAME
    grid.py
DESCRIPTION
    This module consists of the iterator of the voxelized electromagnetic simulation data, so it is in the 3d grid format.
"""
from typing import Union
from pathlib import Path

import numpy as np

from .dataitem import DataItem
from ._base import MagnetBaseIterator


class MagnetGridIterator(MagnetBaseIterator):
    """
    Alias for the iterator of the voxelized electromagnetic simulation data, so it is in the 3d grid format.
    """
    pass
