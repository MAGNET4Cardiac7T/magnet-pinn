"""
    A module containing physics informed losses and constraints for the magnet_pinn package.
"""

from .base import MSELoss, MAELoss, HuberLoss, LogCoshLoss
from .physics import BasePhysicsLoss, DivergenceLoss

__all__ = ['MSELoss', 'MAELoss', 'HuberLoss', 'LogCoshLoss', 'BasePhysicsLoss', 'DivergenceLoss']