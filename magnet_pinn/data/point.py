"""
NAME
    point.py
DESCRIPTION
    This module contains classes for loading the electromagnetic simulation data in the pointscloud format.
"""
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
    """
    pass
