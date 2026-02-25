"""Swin Transformer U-Net architectures for 2D and 3D EM field prediction."""

from .models import SwinTransformerUNet, SwinTransformerUNet3D, SwinTransformerUNet2D

__all__ = [
    "SwinTransformerUNet",
    "SwinTransformerUNet3D",
    "SwinTransformerUNet2D",
]
