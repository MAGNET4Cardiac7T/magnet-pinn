"""Tests for foundational 3D U-Net utilities, building blocks, and squeeze-excitation blocks."""

import pytest
import torch
from torch import nn

from magnet_pinn.models import (
    ResidualUNet2D,
    ResidualUNet3D,
    ResidualUNetSE3D,
    UNet2D,
    UNet3D,
)
from magnet_pinn.models._unet3d.buildingblocks import (
    Decoder,
    DoubleConv,
    Encoder,
    InterpolateUpsampling,
    NoUpsampling,
    ResNetBlock,
    ResNetBlockSE,
    SingleConv,
    TransposeConvUpsampling,
    create_conv,
    create_decoders,
    create_encoders,
)
from magnet_pinn.models._unet3d.models import get_model
from magnet_pinn.models._unet3d.se import (
    ChannelSELayer3D,
    ChannelSpatialSELayer3D,
    SpatialSELayer3D,
)
from magnet_pinn.models._unet3d.utils import get_class, number_of_features_per_level


def _repeat_channels(input_tensor: torch.Tensor, channels: int) -> torch.Tensor:
    """Repeat the fixture along the channel axis to create a multi-channel input."""
    repeats = [1, channels] + [1] * (input_tensor.dim() - 2)
    return input_tensor.repeat(*repeats)


def _expand_to_eight_channels(input_tensor: torch.Tensor) -> torch.Tensor:
    """Repeat the single-channel fixture to create an 8-channel input."""
    return _repeat_channels(input_tensor, 8)


def _assert_nonzero_gradients(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    module: nn.Module | None = None,
) -> None:
    """Backpropagate and assert non-zero input and parameter gradients."""
    output_tensor.sum().backward()

    assert input_tensor.grad is not None
    assert torch.count_nonzero(input_tensor.grad).item() > 0

    if module is not None:
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in module.parameters()
            if parameter.requires_grad
        )


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

        _assert_nonzero_gradients(output_tensor, input_tensor, layer)


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

        _assert_nonzero_gradients(output_tensor, input_tensor, layer)


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

        _assert_nonzero_gradients(output_tensor, input_tensor, layer)


class TestSingleConv:
    @pytest.mark.parametrize("order", ["cr", "gcr", "cl", "ce", "bcr", "cbrd"])
    @pytest.mark.parametrize("is3d", [True, False])
    def test_forward_preserves_spatial_shape_and_uses_expected_bias(
        self,
        order: str,
        is3d: bool,
        small_3d_input: torch.Tensor,
        small_2d_input: torch.Tensor,
    ) -> None:
        input_tensor = small_3d_input if is3d else small_2d_input
        layer = SingleConv(1, 8, order=order, num_groups=8, is3d=is3d)

        output_tensor = layer(input_tensor)

        assert output_tensor.shape == (1, 8, *input_tensor.shape[2:])
        if "g" in order or "b" in order:
            assert layer.conv.bias is None
        else:
            assert layer.conv.bias is not None

    def test_capital_d_order_creates_dropout2d_layer(self, small_2d_input: torch.Tensor) -> None:
        layer = SingleConv(1, 8, order="cbrD", num_groups=8, is3d=False)

        output_tensor = layer(small_2d_input)

        assert isinstance(layer.dropout2d, nn.Dropout2d)
        assert output_tensor.shape == (1, 8, *small_2d_input.shape[2:])


class TestDoubleConv:
    @pytest.mark.parametrize(
        ("upscale", "expected_conv1_out_channels"),
        [(1, 16), (2, 8)],
    )
    def test_encoder_path_uses_expected_intermediate_channels(
        self,
        small_3d_input: torch.Tensor,
        upscale: int,
        expected_conv1_out_channels: int,
    ) -> None:
        module = DoubleConv(8, 16, encoder=True, upscale=upscale, num_groups=8, is3d=True)
        input_tensor = _expand_to_eight_channels(small_3d_input)

        output_tensor = module(input_tensor)

        assert module.SingleConv1.conv.in_channels == 8
        assert module.SingleConv1.conv.out_channels == expected_conv1_out_channels
        assert module.SingleConv2.conv.in_channels == expected_conv1_out_channels
        assert module.SingleConv2.conv.out_channels == 16
        assert output_tensor.shape == (1, 16, *small_3d_input.shape[2:])

    def test_encoder_path_clamps_intermediate_channels_to_input_channels(self) -> None:
        module = DoubleConv(32, 16, encoder=True, upscale=2, num_groups=8, is3d=True)
        input_tensor = torch.randn(1, 32, 8, 8, 8)

        output_tensor = module(input_tensor)

        assert module.SingleConv1.conv.out_channels == 32
        assert module.SingleConv2.conv.in_channels == 32
        assert output_tensor.shape == (1, 16, 8, 8, 8)

    def test_decoder_path_reduces_channels_in_first_convolution(
        self,
        small_3d_input: torch.Tensor,
    ) -> None:
        module = DoubleConv(24, 8, encoder=False, num_groups=8, is3d=True)
        input_tensor = _repeat_channels(small_3d_input, 24)

        output_tensor = module(input_tensor)

        assert module.SingleConv1.conv.in_channels == 24
        assert module.SingleConv1.conv.out_channels == 8
        assert module.SingleConv2.conv.in_channels == 8
        assert module.SingleConv2.conv.out_channels == 8
        assert output_tensor.shape == (1, 8, *small_3d_input.shape[2:])

    def test_tuple_dropout_prob_applies_to_each_convolution(self, small_3d_input: torch.Tensor) -> None:
        module = DoubleConv(
            8,
            16,
            encoder=True,
            order="cbrd",
            num_groups=8,
            dropout_prob=(0.1, 0.2),
            is3d=True,
        )
        input_tensor = _expand_to_eight_channels(small_3d_input)

        output_tensor = module(input_tensor)

        assert output_tensor.shape == (1, 16, *small_3d_input.shape[2:])
        assert isinstance(module.SingleConv1.dropout, nn.Dropout)
        assert isinstance(module.SingleConv2.dropout, nn.Dropout)
        assert module.SingleConv1.dropout.p == pytest.approx(0.1)
        assert module.SingleConv2.dropout.p == pytest.approx(0.2)


class TestResNetBlock:
    @pytest.mark.parametrize(
        ("order", "expected_non_linearity"),
        [("cge", nn.ELU), ("cgr", nn.ReLU), ("cgl", nn.LeakyReLU)],
    )
    def test_identity_shortcut_preserves_shape_and_selects_non_linearity(
        self,
        small_3d_input: torch.Tensor,
        order: str,
        expected_non_linearity: type[nn.Module],
    ) -> None:
        block = ResNetBlock(8, 8, order=order, num_groups=8, is3d=True)
        input_tensor = _expand_to_eight_channels(small_3d_input).clone().detach().requires_grad_(True)

        output_tensor = block(input_tensor)

        assert isinstance(block.conv1, nn.Identity)
        assert isinstance(block.non_linearity, expected_non_linearity)
        assert output_tensor.shape == input_tensor.shape
        _assert_nonzero_gradients(output_tensor, input_tensor, block)

    @pytest.mark.parametrize(
        ("is3d", "expected_projection_type"),
        [(True, nn.Conv3d), (False, nn.Conv2d)],
    )
    def test_projection_shortcut_uses_convolution_when_channels_change(
        self,
        is3d: bool,
        expected_projection_type: type[nn.Module],
        small_3d_input: torch.Tensor,
        small_2d_input: torch.Tensor,
    ) -> None:
        block = ResNetBlock(1, 8, order="cge", num_groups=8, is3d=is3d)
        input_tensor = (small_3d_input if is3d else small_2d_input).clone().detach().requires_grad_(True)

        output_tensor = block(input_tensor)

        assert isinstance(block.conv1, expected_projection_type)
        assert output_tensor.shape == (1, 8, *input_tensor.shape[2:])
        _assert_nonzero_gradients(output_tensor, input_tensor, block)


class TestResNetBlockSE:
    @pytest.mark.parametrize(
        ("se_module", "expected_se_type"),
        [
            ("scse", ChannelSpatialSELayer3D),
            ("cse", ChannelSELayer3D),
            ("sse", SpatialSELayer3D),
        ],
    )
    def test_forward_preserves_shape_and_gradients(
        self,
        small_3d_input: torch.Tensor,
        se_module: str,
        expected_se_type: type[nn.Module],
    ) -> None:
        block = ResNetBlockSE(8, 8, num_groups=8, se_module=se_module)
        input_tensor = _expand_to_eight_channels(small_3d_input).clone().detach().requires_grad_(True)

        output_tensor = block(input_tensor)

        assert isinstance(block.se_module, expected_se_type)
        assert output_tensor.shape == input_tensor.shape
        _assert_nonzero_gradients(output_tensor, input_tensor, block)

    def test_invalid_se_module_raises_assertion(self) -> None:
        with pytest.raises(AssertionError):
            ResNetBlockSE(8, 8, num_groups=8, se_module="invalid")


class TestEncoder:
    @pytest.mark.parametrize(
        ("pool_type", "expected_pool_type"),
        [("max", nn.MaxPool3d), ("avg", nn.AvgPool3d)],
    )
    def test_pooling_halves_spatial_dimensions(
        self,
        small_3d_input: torch.Tensor,
        pool_type: str,
        expected_pool_type: type[nn.Module],
    ) -> None:
        encoder = Encoder(
            1,
            8,
            apply_pooling=True,
            pool_type=pool_type,
            basic_module=DoubleConv,
            num_groups=8,
            is3d=True,
        )

        output_tensor = encoder(small_3d_input)

        assert isinstance(encoder.pooling, expected_pool_type)
        assert output_tensor.shape == (1, 8, 4, 4, 4)

    def test_without_pooling_preserves_spatial_dimensions(self, small_3d_input: torch.Tensor) -> None:
        encoder = Encoder(
            1,
            8,
            apply_pooling=False,
            basic_module=DoubleConv,
            num_groups=8,
            is3d=True,
        )

        output_tensor = encoder(small_3d_input)

        assert encoder.pooling is None
        assert output_tensor.shape == (1, 8, *small_3d_input.shape[2:])

    def test_resnet_block_encoder_supports_2d_inputs(self, small_2d_input: torch.Tensor) -> None:
        encoder = Encoder(
            1,
            8,
            apply_pooling=True,
            pool_type="max",
            basic_module=ResNetBlock,
            conv_layer_order="cge",
            num_groups=8,
            is3d=False,
        )

        output_tensor = encoder(small_2d_input)

        assert isinstance(encoder.pooling, nn.MaxPool2d)
        assert isinstance(encoder.basic_module, ResNetBlock)
        assert output_tensor.shape == (1, 8, 8, 8)

    def test_invalid_pool_type_raises_assertion(self) -> None:
        with pytest.raises(AssertionError):
            Encoder(1, 8, pool_type="median", basic_module=DoubleConv, num_groups=8, is3d=True)

    def test_create_encoders_builds_first_level_without_pooling(
        self,
        small_3d_input: torch.Tensor,
    ) -> None:
        encoders = create_encoders(
            in_channels=1,
            f_maps=[8, 16],
            basic_module=DoubleConv,
            conv_kernel_size=3,
            conv_padding=1,
            conv_upscale=2,
            dropout_prob=0.1,
            layer_order="gcr",
            num_groups=8,
            pool_kernel_size=2,
            is3d=True,
        )

        first_output = encoders[0](small_3d_input)
        second_output = encoders[1](first_output)

        assert len(encoders) == 2
        assert encoders[0].pooling is None
        assert isinstance(encoders[1].pooling, nn.MaxPool3d)
        assert first_output.shape == (1, 8, 8, 8, 8)
        assert second_output.shape == (1, 16, 4, 4, 4)


class TestDecoder:
    def test_double_conv_default_uses_interpolation_and_concat(
        self,
        small_3d_input: torch.Tensor,
    ) -> None:
        encoder_features = _expand_to_eight_channels(small_3d_input)
        x = torch.randn(1, 16, 4, 4, 4)
        decoder = Decoder(
            24,
            8,
            basic_module=DoubleConv,
            num_groups=8,
            upsample="default",
            is3d=True,
        )

        output_tensor = decoder(encoder_features, x)

        assert isinstance(decoder.upsampling, InterpolateUpsampling)
        assert decoder.joining.keywords == {"concat": True}
        assert decoder.basic_module.SingleConv1.conv.in_channels == 24
        assert output_tensor.shape == encoder_features.shape

    def test_resnet_default_uses_transposed_convolution_and_sum_joining(
        self,
        small_3d_input: torch.Tensor,
    ) -> None:
        encoder_features = _expand_to_eight_channels(small_3d_input)
        x = torch.randn(1, 16, 4, 4, 4)
        decoder = Decoder(
            16,
            8,
            basic_module=ResNetBlock,
            conv_layer_order="cge",
            num_groups=8,
            upsample="default",
            is3d=True,
        )

        output_tensor = decoder(encoder_features, x)

        assert isinstance(decoder.upsampling, TransposeConvUpsampling)
        assert decoder.joining.keywords == {"concat": False}
        assert isinstance(decoder.upsampling.upsample.conv_transposed, nn.ConvTranspose3d)
        assert isinstance(decoder.basic_module.conv1, nn.Identity)
        assert decoder.basic_module.conv2.conv.in_channels == 8
        assert output_tensor.shape == encoder_features.shape

    @pytest.mark.parametrize("upsample", [None, "none"])
    def test_none_upsampling_uses_no_upsampling(
        self,
        small_3d_input: torch.Tensor,
        upsample: str | None,
    ) -> None:
        encoder_features = _expand_to_eight_channels(small_3d_input)
        x = encoder_features.clone()
        decoder = Decoder(
            16,
            8,
            basic_module=DoubleConv,
            num_groups=8,
            upsample=upsample,
            is3d=True,
        )

        output_tensor = decoder(encoder_features, x)

        assert isinstance(decoder.upsampling, NoUpsampling)
        assert decoder.joining.keywords == {"concat": True}
        assert decoder.basic_module.SingleConv1.conv.in_channels == 16
        assert output_tensor.shape == encoder_features.shape

    def test_create_decoders_uses_concat_channel_count_for_double_conv(self) -> None:
        decoders = create_decoders(
            f_maps=[8, 16],
            basic_module=DoubleConv,
            conv_kernel_size=3,
            conv_padding=1,
            layer_order="gcr",
            num_groups=8,
            upsample="default",
            dropout_prob=0.1,
            is3d=True,
        )

        assert len(decoders) == 1
        assert isinstance(decoders[0].upsampling, InterpolateUpsampling)
        assert decoders[0].basic_module.SingleConv1.conv.in_channels == 24

    def test_create_decoders_uses_sum_channel_count_for_residual_blocks(self) -> None:
        decoders = create_decoders(
            f_maps=[8, 16],
            basic_module=ResNetBlock,
            conv_kernel_size=3,
            conv_padding=1,
            layer_order="cge",
            num_groups=8,
            upsample="default",
            dropout_prob=0.1,
            is3d=True,
        )

        assert len(decoders) == 1
        assert isinstance(decoders[0].upsampling, TransposeConvUpsampling)
        assert decoders[0].upsampling.upsample.conv_transposed.in_channels == 16
        assert decoders[0].joining.keywords == {"concat": False}


class TestUpsamplingModules:
    def test_interpolate_upsampling_matches_encoder_spatial_size(
        self,
        small_3d_input: torch.Tensor,
    ) -> None:
        upsampling = InterpolateUpsampling(mode="nearest")
        encoder_features = _expand_to_eight_channels(small_3d_input)
        x = torch.randn(1, 16, 4, 4, 4)

        output_tensor = upsampling(encoder_features=encoder_features, x=x)

        assert output_tensor.shape == (1, 16, 8, 8, 8)

    @pytest.mark.parametrize(
        ("is3d", "encoder_shape", "decoder_shape", "expected_conv_type"),
        [
            (True, (1, 8, 8, 8, 8), (1, 16, 4, 4, 4), nn.ConvTranspose3d),
            (False, (1, 8, 16, 16), (1, 16, 8, 8), nn.ConvTranspose2d),
        ],
    )
    def test_transpose_conv_upsampling_matches_encoder_spatial_size(
        self,
        is3d: bool,
        encoder_shape: tuple[int, ...],
        decoder_shape: tuple[int, ...],
        expected_conv_type: type[nn.Module],
    ) -> None:
        upsampling = TransposeConvUpsampling(16, 8, is3d=is3d)
        encoder_features = torch.randn(*encoder_shape)
        x = torch.randn(*decoder_shape)

        output_tensor = upsampling(encoder_features=encoder_features, x=x)

        assert isinstance(upsampling.upsample.conv_transposed, expected_conv_type)
        assert output_tensor.shape == encoder_features.shape

    def test_no_upsampling_returns_input_tensor_unchanged(self, small_3d_input: torch.Tensor) -> None:
        upsampling = NoUpsampling()
        encoder_features = _expand_to_eight_channels(small_3d_input)
        x = encoder_features.clone()

        output_tensor = upsampling(encoder_features=encoder_features, x=x)

        assert output_tensor is x
        assert output_tensor.shape == x.shape


class TestCreateConvErrors:
    def test_missing_convolution_layer_raises_assertion(self) -> None:
        with pytest.raises(AssertionError, match="Conv layer MUST be present"):
            create_conv(1, 8, 3, "gr", 8, 1, 0.1, True)

    def test_non_linearity_first_raises_assertion(self) -> None:
        with pytest.raises(AssertionError, match="Non-linearity cannot be the first operation"):
            create_conv(1, 8, 3, "rc", 8, 1, 0.1, True)

    def test_unsupported_layer_character_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported layer type 'z'"):
            create_conv(1, 8, 3, "cz", 8, 1, 0.1, True)

    def test_groupnorm_clamps_num_groups_to_one_when_channels_are_smaller(self) -> None:
        modules = create_conv(4, 8, 3, "gcr", 8, 1, 0.1, True)
        module_name, groupnorm = modules[0]

        assert module_name == "groupnorm"
        assert isinstance(groupnorm, nn.GroupNorm)
        assert groupnorm.num_groups == 1
        assert groupnorm.num_channels == 4


class TestUNetModels:
    @pytest.mark.parametrize(
        ("model_class", "is3d"),
        [
            pytest.param(UNet3D, True, id="unet3d"),
            pytest.param(ResidualUNet3D, True, id="residual-unet3d"),
            pytest.param(ResidualUNetSE3D, True, id="residual-unetse3d"),
            pytest.param(UNet2D, False, id="unet2d"),
            pytest.param(ResidualUNet2D, False, id="residual-unet2d"),
        ],
    )
    def test_forward_preserves_shape_and_gradients(
        self,
        model_class: type[nn.Module],
        is3d: bool,
        small_3d_input: torch.Tensor,
        small_2d_input: torch.Tensor,
    ) -> None:
        model = model_class(1, 2, f_maps=8, num_levels=2, num_groups=8)
        input_tensor = (small_3d_input if is3d else small_2d_input).clone().detach().requires_grad_(True)

        output_tensor = model(input_tensor)

        assert output_tensor.shape == (1, 2, *input_tensor.shape[2:])
        _assert_nonzero_gradients(output_tensor, input_tensor, model)


class TestUNetFMapsVariants:
    @pytest.mark.parametrize(
        "f_maps",
        [pytest.param([8, 16], id="list"), pytest.param((8, 16), id="tuple")],
    )
    def test_explicit_f_maps_sequences_preserve_forward_shape_and_gradients(
        self,
        small_3d_input: torch.Tensor,
        f_maps: list[int] | tuple[int, int],
    ) -> None:
        model = UNet3D(1, 2, f_maps=f_maps, num_groups=8)
        input_tensor = small_3d_input.clone().detach().requires_grad_(True)

        output_tensor = model(input_tensor)

        assert output_tensor.shape == (1, 2, *small_3d_input.shape[2:])
        _assert_nonzero_gradients(output_tensor, input_tensor, model)


class TestAbstractUNetAssertions:
    def test_single_level_f_maps_raises_assertion(self) -> None:
        with pytest.raises(AssertionError, match="Required at least 2 levels in the U-Net"):
            UNet3D(1, 2, f_maps=[8])

    def test_groupnorm_layer_order_requires_num_groups(self) -> None:
        with pytest.raises(AssertionError, match="num_groups must be specified if GroupNorm is used"):
            UNet3D(1, 2, layer_order="gcr", num_groups=None, f_maps=[8, 16])


class TestGetModel:
    def test_missing_external_pytorch3dunet_dependency_raises_module_not_found_error(self) -> None:
        with pytest.raises(ModuleNotFoundError, match="No module named 'pytorch3dunet'"):
            get_model({"name": "UNet3D"})
