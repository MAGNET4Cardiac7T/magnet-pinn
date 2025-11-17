"""
Physics informed losses and constraints for the magnet_pinn package.
"""

from .base import MSELoss, MAELoss, HuberLoss, LogCoshLoss
from .physics import BasePhysicsLoss, DivergenceLoss, FaradaysLoss

__all__ = [
    'MSELoss',
    'MAELoss',
    'HuberLoss',
    'LogCoshLoss',
    'BasePhysicsLoss',
    'DivergenceLoss',
    'FaradaysLoss',
]
