"""Module contains data structures for simulation"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Simulation:
    name: str
    path: str
    e_field: Optional[np.array] = None
    h_field: Optional[np.array] = None
    object_masks: Optional[np.array] = None
    coordinates: Optional[np.array] = None
    features: Optional[np.array] = None
