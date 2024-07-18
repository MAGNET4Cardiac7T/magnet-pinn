"""
NAME 
    simulation.py

DESCRIPTION
    This module contains the Simulation dataclass, 
    which is used to store the data of a simulation.

CLASSES
    Simulation
"""
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
    general_mask: Optional[np.array] = None
