"""Pytest fixtures scoped to model architecture tests."""

from collections.abc import Generator

import pytest
import torch


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
