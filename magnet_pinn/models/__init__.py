"""
    A module containing models for predicting EM fields in a MRI scanner.
"""

from ._unet3d.models import UNet3D, ResidualUNet3D, ResidualUNetSE3D, UNet2D, ResidualUNet2D, AbstractUNet
from ._swin_transformer.models import SwinTransformerUNet, SwinTransformerUNet3D, SwinTransformerUNet2D
from ._transsolver.models import Transolver, Transolver3D, Transolver2D

__all__ = [
    "AbstractUNet",
    "UNet3D",
    "ResidualUNet3D",
    "ResidualUNetSE3D",
    "UNet2D",
    "ResidualUNet2D",
    "SwinTransformerUNet",
    "SwinTransformerUNet3D",
    "SwinTransformerUNet2D",
    "Transolver",
    "Transolver3D",
    "Transolver2D",
]