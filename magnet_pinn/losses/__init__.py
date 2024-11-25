"""
    This module contains physics informed losses and constraints for the magnet_pinn package.
"""

from .base import MSELoss, MAELoss, HuberLoss, LogCoshLoss

__all__ = ['MSELoss', 'MAELoss', 'HuberLoss', 'LogCoshLoss']