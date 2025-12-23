"""NAME
    point.py
DESCRIPTION
    This module contains classes for loading the electromagnetic simulation data in the pointscloud format.
"""
import warnings
from typing import Union
from pathlib import Path

import h5py
import numpy as np

from .dataitem import DataItem
from ._base import MagnetBaseIterator
from magnet_pinn.preprocessing.preprocessing import COORDINATES_OUT_KEY


class MagnetPointIterator(MagnetBaseIterator):
    """
    Alias for the iterator of the electromagnetic simulation data in the pointscloud format.
    A class is deprecated and will be removed.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MagnetPointIterator is deprecated and will be removed. "
            "Use MagnetBaseIterator directly instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
