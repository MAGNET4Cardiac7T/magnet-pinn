"""
Physics informed losses and constraints for the magnet_pinn package.
"""

from .base import MSELoss, MAELoss, HuberLoss, LogCoshLoss
from .physics import (
    BasePhysicsLoss,
    DivergenceLoss,
    FaradaysLawLoss,
    MRI_FREQUENCY_HZ,
    VACUUM_PERMEABILITY,
)

__all__ = [
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "LogCoshLoss",
    "BasePhysicsLoss",
    "DivergenceLoss",
    "FaradaysLawLoss",
    "MRI_FREQUENCY_HZ",
    "VACUUM_PERMEABILITY",
]
