"""Shared pytest fixtures for model tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest
import torch

try:
    from magnet_pinn.models import (  # type: ignore[attr-defined]
        SwinTransformerUNet2D,
        SwinTransformerUNet3D,
    )

    _swin_available = True
except ImportError:
    _swin_available = False

_swin_missing = pytest.mark.skipif(
    not _swin_available, reason="SwinTransformerUNet not yet available in this build"
)


@pytest.fixture(autouse=True)
def torch_deterministic() -> Generator[None, None, None]:
    """Set a deterministic Torch seed for every model test."""
    torch.manual_seed(42)
    yield


@pytest.fixture
def small_3d_input() -> torch.Tensor:
    """Return a small 3D tensor for lightweight model tests."""
    return torch.randn(1, 1, 8, 8, 8)


@pytest.fixture
def small_2d_input() -> torch.Tensor:
    """Return a small 2D tensor for lightweight model tests."""
    return torch.randn(1, 1, 16, 16)


@pytest.fixture(scope="module")
def batch_size() -> int:
    return 2


@pytest.fixture(scope="module")
def in_channels() -> int:
    return 3


@pytest.fixture(scope="module")
def out_channels() -> int:
    return 6


@pytest.fixture(scope="module")
def spatial_size_3d() -> int:
    """
    3D spatial size aligned to the model's padding divisor.

    With patch_size=2, window_size=2, num_stages=2:
        divisor = patch_size * window_size * 2^(num_stages-1) = 2 * 2 * 2 = 8
    """
    return 16


@pytest.fixture(scope="module")
def spatial_size_2d() -> int:
    """
    2D spatial size aligned to the model's padding divisor.

    With patch_size=2, window_size=2, num_stages=2:
        divisor = 2 * 2 * 2 = 8
    """
    return 16


@_swin_missing
@pytest.fixture(scope="module")
def swin3d_model(
    in_channels: int, out_channels: int
) -> "SwinTransformerUNet3D":
    """
    Minimal SwinTransformerUNet3D for shape and gradient tests.

    Uses 2 stages, patch_size=2, window_size=2 and embed_dim=8 to keep the
    model small and the spatial divisor manageable (8 voxels).
    """
    return SwinTransformerUNet3D(  # type: ignore[name-defined]
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


@_swin_missing
@pytest.fixture(scope="module")
def swin2d_model(
    in_channels: int, out_channels: int
) -> "SwinTransformerUNet2D":
    """
    Minimal SwinTransformerUNet2D for shape and gradient tests.

    Uses 2 stages, patch_size=2, window_size=2 and embed_dim=8 to keep the
    model small and the spatial divisor manageable (8 pixels).
    """
    return SwinTransformerUNet2D(  # type: ignore[name-defined]
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
