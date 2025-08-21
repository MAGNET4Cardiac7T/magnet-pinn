"""
    A module containing models for predicting EM fields in a MRI scanner.
"""

from ._unet3d.models import UNet3D

__all__ = [
    "UNet3D",
]