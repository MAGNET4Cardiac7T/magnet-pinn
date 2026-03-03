"""Shared pytest fixtures for model tests."""

import pytest
import torch

from magnet_pinn.models import SwinTransformerUNet3D, SwinTransformerUNet2D
from magnet_pinn.models import Transolver3D, Transolver2D


@pytest.fixture(scope="module")
def batch_size():
    return 2


@pytest.fixture(scope="module")
def in_channels():
    return 3


@pytest.fixture(scope="module")
def out_channels():
    return 6


@pytest.fixture(scope="module")
def spatial_size_3d():
    """
    3D spatial size aligned to the model's padding divisor.

    With patch_size=2, window_size=2, num_stages=2:
        divisor = patch_size * window_size * 2^(num_stages-1) = 2 * 2 * 2 = 8
    """
    return 16   # 2 * 8


@pytest.fixture(scope="module")
def spatial_size_2d():
    """
    2D spatial size aligned to the model's padding divisor.

    With patch_size=2, window_size=2, num_stages=2:
        divisor = 2 * 2 * 2 = 8
    """
    return 16


@pytest.fixture(scope="module")
def swin3d_model(in_channels, out_channels):
    """
    Minimal SwinTransformerUNet3D for shape and gradient tests.

    Uses 2 stages, patch_size=2, window_size=2 and embed_dim=8 to keep the
    model small and the spatial divisor manageable (8 voxels).
    """
    return SwinTransformerUNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=2,
        embed_dim=8,
        depths=[1, 1],
        num_heads=[2, 4],
        window_size=2,
        dropout_prob=0.0,
        attn_dropout=0.0,
    )


@pytest.fixture(scope="module")
def swin2d_model(in_channels, out_channels):
    """
    Minimal SwinTransformerUNet2D for shape and gradient tests.

    Uses 2 stages, patch_size=2, window_size=2 and embed_dim=8 to keep the
    model small and the spatial divisor manageable (8 pixels).
    """
    return SwinTransformerUNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=2,
        embed_dim=8,
        depths=[1, 1],
        num_heads=[2, 4],
        window_size=2,
        dropout_prob=0.0,
        attn_dropout=0.0,
    )


@pytest.fixture(scope="module")
def transolver3d_model(in_channels, out_channels):
    """
    Minimal Transolver3D for shape and gradient tests.

    Uses embed_dim=8, depth=1, n_slices=4, num_heads=2 to keep the model
    small and fast. No padding requirement, so any spatial size works.
    """
    return Transolver3D(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=8,
        depth=1,
        n_slices=4,
        num_heads=2,
        mlp_ratio=2.0,
        dropout_prob=0.0,
        attn_dropout=0.0,
    )


@pytest.fixture(scope="module")
def transolver2d_model(in_channels, out_channels):
    """
    Minimal Transolver2D for shape and gradient tests.

    Uses embed_dim=8, depth=1, n_slices=4, num_heads=2 to keep the model
    small and fast. No padding requirement, so any spatial size works.
    """
    return Transolver2D(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=8,
        depth=1,
        n_slices=4,
        num_heads=2,
        mlp_ratio=2.0,
        dropout_prob=0.0,
        attn_dropout=0.0,
    )
