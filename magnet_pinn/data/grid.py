"""NAME
    grid.py
DESCRIPTION
    This module consists of the iterator of the voxelized electromagnetic simulation data, so it is in the 3d grid format.
"""
import warnings
from typing import Union
from pathlib import Path

import numpy as np

from .dataitem import DataItem
from ._base import MagnetBaseIterator


class MagnetGridIterator(MagnetBaseIterator):
    """
    Alias for the iterator of the voxelized electromagnetic simulation data, so it is in the 3d grid format.
    A class is deprecated and will be removed.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MagnetGridIterator is deprecated and will be removed. "
            "Use MagnetBaseIterator directly instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
