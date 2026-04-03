"""Tests for foundational 3D U-Net utilities and squeeze-excitation blocks."""

import pytest
import torch
from torch import nn

from magnet_pinn.models._unet3d.se import (
    ChannelSELayer3D,
    ChannelSpatialSELayer3D,
    SpatialSELayer3D,
)
from magnet_pinn.models._unet3d.utils import get_class, number_of_features_per_level


def _expand_to_eight_channels(input_tensor: torch.Tensor) -> torch.Tensor:
    """Repeat the single-channel fixture to create an 8-channel 3D input."""
    return input_tensor.repeat(1, 8, 1, 1, 1)


def _assert_nonzero_input_gradients(output_tensor: torch.Tensor, input_tensor: torch.Tensor) -> None:
    """Backpropagate through a layer output and assert non-zero input gradients."""
    output_tensor.sum().backward()

    assert input_tensor.grad is not None
    assert torch.count_nonzero(input_tensor.grad).item() > 0


class TestUtils:
    def test_get_class_returns_requested_class(self) -> None:
        loaded_class = get_class("ReLU", ["torch.nn"])

        assert loaded_class is nn.ReLU

    def test_get_class_raises_for_unknown_class(self) -> None:
        with pytest.raises(RuntimeError, match="Unsupported dataset class"):
            get_class("MissingClass", ["torch.nn"])

    def test_number_of_features_per_level_doubles_per_level(self) -> None:
        assert number_of_features_per_level(8, 3) == [8, 16, 32]


class TestChannelSELayer3D:
    @pytest.mark.parametrize("reduction_ratio", [1, 2, 4])
    def test_forward_preserves_shape(
        self,
        small_3d_input: torch.Tensor,
        reduction_ratio: int,
    ) -> None:
        layer = ChannelSELayer3D(num_channels=8, reduction_ratio=reduction_ratio)
        input_tensor = _expand_to_eight_channels(small_3d_input)

        output_tensor = layer(input_tensor)

        assert output_tensor.shape == input_tensor.shape

    @pytest.mark.parametrize("reduction_ratio", [1, 2, 4])
    def test_backward_produces_nonzero_input_gradients(
        self,
        small_3d_input: torch.Tensor,
        reduction_ratio: int,
    ) -> None:
        layer = ChannelSELayer3D(num_channels=8, reduction_ratio=reduction_ratio)
        input_tensor = _expand_to_eight_channels(small_3d_input).clone().detach().requires_grad_(True)

        output_tensor = layer(input_tensor)

        _assert_nonzero_input_gradients(output_tensor, input_tensor)


class TestSpatialSELayer3D:
    """Only test the default weights=None path because the weights branch is dead code for 5D inputs."""

    def test_forward_preserves_shape(self, small_3d_input: torch.Tensor) -> None:
        layer = SpatialSELayer3D(num_channels=8)
        input_tensor = _expand_to_eight_channels(small_3d_input)

        output_tensor = layer(input_tensor)

        assert output_tensor.shape == input_tensor.shape

    def test_backward_produces_nonzero_input_gradients(self, small_3d_input: torch.Tensor) -> None:
        layer = SpatialSELayer3D(num_channels=8)
        input_tensor = _expand_to_eight_channels(small_3d_input).clone().detach().requires_grad_(True)

        output_tensor = layer(input_tensor)

        _assert_nonzero_input_gradients(output_tensor, input_tensor)


class TestChannelSpatialSELayer3D:
    @pytest.mark.parametrize("reduction_ratio", [1, 2])
    def test_forward_preserves_shape(
        self,
        small_3d_input: torch.Tensor,
        reduction_ratio: int,
    ) -> None:
        layer = ChannelSpatialSELayer3D(num_channels=8, reduction_ratio=reduction_ratio)
        input_tensor = _expand_to_eight_channels(small_3d_input)

        output_tensor = layer(input_tensor)

        assert output_tensor.shape == input_tensor.shape

    @pytest.mark.parametrize("reduction_ratio", [1, 2])
    def test_backward_produces_nonzero_input_gradients(
        self,
        small_3d_input: torch.Tensor,
        reduction_ratio: int,
    ) -> None:
        layer = ChannelSpatialSELayer3D(num_channels=8, reduction_ratio=reduction_ratio)
        input_tensor = _expand_to_eight_channels(small_3d_input).clone().detach().requires_grad_(True)

        output_tensor = layer(input_tensor)

        _assert_nonzero_input_gradients(output_tensor, input_tensor)
