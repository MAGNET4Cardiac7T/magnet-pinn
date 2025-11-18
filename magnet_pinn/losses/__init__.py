"""
Physics informed losses and constraints for the magnet_pinn package.
"""

from .base import MSELoss, MAELoss, HuberLoss, LogCoshLoss
from .physics import BasePhysicsLoss, DivergenceLoss, FaradaysLoss, MRI_FREQUENCY_HZ, VACUUM_PERMEABILITY
from .utils import mask_padding

__all__ = [
    'MSELoss',
    'MAELoss',
    'HuberLoss',
    'LogCoshLoss',
    'BasePhysicsLoss',
    'DivergenceLoss',
    'FaradaysLoss',
    'MRI_FREQUENCY_HZ',
    'VACUUM_PERMEABILITY',
    'mask_padding',
]
