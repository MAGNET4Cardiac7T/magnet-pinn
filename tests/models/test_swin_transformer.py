"""Tests for the Swin Transformer U-Net implementation."""

import pytest
import torch

from magnet_pinn.models._swin_transformer.buildingblocks import (
    PatchExpanding,
    PatchMerging,
    SwinTransformerBlock,
    WindowAttention,
    window_partition,
    window_partition_2d,
    window_reverse,
    window_reverse_2d,
)


# ---------------------------------------------------------------------------
# Output shape — 3D
# ---------------------------------------------------------------------------

class TestOutputShape3D:
    def test_aligned_input(self, batch_size, in_channels, out_channels,
                           spatial_size_3d, swin3d_model):
        """Model returns same spatial dims as input when size is grid-aligned."""
        s = spatial_size_3d
        x = torch.randn(batch_size, in_channels, s, s, s)
        with torch.no_grad():
            y = swin3d_model(x)
        assert y.shape == (batch_size, out_channels, s, s, s)

    def test_arbitrary_input(self, batch_size, in_channels, out_channels, swin3d_model):
        """Model pads, processes, and crops arbitrary (non-aligned) spatial sizes."""
        x = torch.randn(batch_size, in_channels, 13, 15, 11)
        with torch.no_grad():
            y = swin3d_model(x)
        assert y.shape == (batch_size, out_channels, 13, 15, 11)

    def test_non_cubic_input(self, batch_size, in_channels, out_channels, swin3d_model):
        """Model handles non-cubic inputs with different extents per dimension."""
        x = torch.randn(batch_size, in_channels, 8, 16, 24)
        with torch.no_grad():
            y = swin3d_model(x)
        assert y.shape == (batch_size, out_channels, 8, 16, 24)

    def test_single_batch(self, in_channels, out_channels, swin3d_model):
        """Model works with batch_size=1."""
        x = torch.randn(1, in_channels, 16, 16, 16)
        with torch.no_grad():
            y = swin3d_model(x)
        assert y.shape == (1, out_channels, 16, 16, 16)


# ---------------------------------------------------------------------------
# Output shape — 2D
# ---------------------------------------------------------------------------

class TestOutputShape2D:
    def test_aligned_input(self, batch_size, in_channels, out_channels,
                           spatial_size_2d, swin2d_model):
        """2D model returns same spatial dims for aligned input."""
        s = spatial_size_2d
        x = torch.randn(batch_size, in_channels, s, s)
        with torch.no_grad():
            y = swin2d_model(x)
        assert y.shape == (batch_size, out_channels, s, s)

    def test_arbitrary_input(self, batch_size, in_channels, out_channels, swin2d_model):
        """2D model handles non-aligned spatial sizes."""
        x = torch.randn(batch_size, in_channels, 13, 17)
        with torch.no_grad():
            y = swin2d_model(x)
        assert y.shape == (batch_size, out_channels, 13, 17)

    def test_non_square_input(self, batch_size, in_channels, out_channels, swin2d_model):
        """2D model handles non-square aligned inputs."""
        x = torch.randn(batch_size, in_channels, 8, 16)
        with torch.no_grad():
            y = swin2d_model(x)
        assert y.shape == (batch_size, out_channels, 8, 16)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_3d_gradients_not_nan(self, batch_size, in_channels, swin3d_model):
        """All parameters receive non-NaN gradients from a scalar loss."""
        x = torch.randn(batch_size, in_channels, 16, 16, 16)
        swin3d_model.train()
        y = swin3d_model(x)
        y.mean().backward()
        for name, p in swin3d_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for parameter: {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient for parameter: {name}"

    def test_2d_gradients_not_nan(self, batch_size, in_channels, swin2d_model):
        """2D model: all parameters receive non-NaN gradients."""
        x = torch.randn(batch_size, in_channels, 16, 16)
        swin2d_model.train()
        y = swin2d_model(x)
        y.mean().backward()
        for name, p in swin2d_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for parameter: {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient for parameter: {name}"


# ---------------------------------------------------------------------------
# Determinism in eval mode
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_3d_deterministic_in_eval(self, batch_size, in_channels, swin3d_model):
        """Two consecutive forward passes in eval mode produce identical outputs."""
        x = torch.randn(batch_size, in_channels, 16, 16, 16)
        swin3d_model.eval()
        with torch.no_grad():
            y1 = swin3d_model(x)
            y2 = swin3d_model(x)
        assert torch.allclose(y1, y2)

    def test_2d_deterministic_in_eval(self, batch_size, in_channels, swin2d_model):
        """2D model: two eval forward passes are identical."""
        x = torch.randn(batch_size, in_channels, 16, 16)
        swin2d_model.eval()
        with torch.no_grad():
            y1 = swin2d_model(x)
            y2 = swin2d_model(x)
        assert torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# Window partition roundtrip
# ---------------------------------------------------------------------------

class TestWindowPartition:
    def test_3d_roundtrip(self):
        """window_partition followed by window_reverse is an identity."""
        B, D, H, W, C = 2, 8, 8, 8, 16
        x = torch.randn(B, D, H, W, C)
        ws = 2
        windows = window_partition(x, ws)
        x_rec = window_reverse(windows, ws, D, H, W)
        assert torch.allclose(x, x_rec)

    def test_2d_roundtrip(self):
        """window_partition_2d followed by window_reverse_2d is an identity."""
        B, H, W, C = 2, 8, 8, 16
        x = torch.randn(B, H, W, C)
        ws = 2
        windows = window_partition_2d(x, ws)
        x_rec = window_reverse_2d(windows, ws, H, W)
        assert torch.allclose(x, x_rec)

    def test_3d_window_count(self):
        """window_partition produces the expected number of windows."""
        B, D, H, W, C = 2, 4, 8, 4, 8
        ws = 2
        windows = window_partition(torch.zeros(B, D, H, W, C), ws)
        expected_nW = (D // ws) * (H // ws) * (W // ws)
        assert windows.shape[0] == B * expected_nW

    def test_2d_window_count(self):
        """window_partition_2d produces the expected number of windows."""
        B, H, W, C = 3, 4, 8, 8
        ws = 2
        windows = window_partition_2d(torch.zeros(B, H, W, C), ws)
        expected_nW = (H // ws) * (W // ws)
        assert windows.shape[0] == B * expected_nW


# ---------------------------------------------------------------------------
# PatchMerging
# ---------------------------------------------------------------------------

class TestPatchMerging:
    def test_3d_output_shape(self):
        """3D PatchMerging halves spatial dims and doubles channels."""
        x = torch.randn(2, 32, 8, 8, 8)
        pm = PatchMerging(32, is3d=True)
        y = pm(x)
        assert y.shape == (2, 64, 4, 4, 4)

    def test_2d_output_shape(self):
        """2D PatchMerging halves spatial dims and doubles channels."""
        x = torch.randn(2, 32, 8, 8)
        pm = PatchMerging(32, is3d=False)
        y = pm(x)
        assert y.shape == (2, 64, 4, 4)

    def test_3d_non_cubic(self):
        """3D PatchMerging handles non-cubic feature maps."""
        x = torch.randn(2, 16, 4, 8, 16)
        pm = PatchMerging(16, is3d=True)
        y = pm(x)
        assert y.shape == (2, 32, 2, 4, 8)


# ---------------------------------------------------------------------------
# PatchExpanding
# ---------------------------------------------------------------------------

class TestPatchExpanding:
    def test_3d_output_shape(self):
        """3D PatchExpanding doubles spatial dims and halves channels."""
        x = torch.randn(2, 32, 4, 4, 4)
        pe = PatchExpanding(32, is3d=True)
        y = pe(x)
        assert y.shape == (2, 16, 8, 8, 8)

    def test_2d_output_shape(self):
        """2D PatchExpanding doubles spatial dims and halves channels."""
        x = torch.randn(2, 32, 4, 4)
        pe = PatchExpanding(32, is3d=False)
        y = pe(x)
        assert y.shape == (2, 16, 8, 8)

    def test_3d_spatial_inverse_of_merging(self):
        """PatchExpanding restores the spatial dimensions reduced by PatchMerging."""
        x = torch.randn(2, 32, 8, 8, 8)
        merged = PatchMerging(32, is3d=True)(x)            # (2, 64, 4, 4, 4)
        expanded = PatchExpanding(64, is3d=True)(merged)   # (2, 32, 8, 8, 8)
        assert expanded.shape == x.shape

    def test_odd_channel_raises(self):
        """PatchExpanding must raise AssertionError for odd channel counts."""
        with pytest.raises(AssertionError):
            PatchExpanding(97, is3d=True)


# ---------------------------------------------------------------------------
# WindowAttention
# ---------------------------------------------------------------------------

class TestWindowAttention:
    def test_3d_output_shape(self):
        """3D WindowAttention preserves token sequence shape."""
        ws = 2
        N = ws ** 3
        nW, B = 4, 2
        dim = 16
        attn = WindowAttention(dim, ws, num_heads=2, is3d=True)
        x = torch.randn(nW * B, N, dim)
        y = attn(x)
        assert y.shape == x.shape

    def test_2d_output_shape(self):
        """2D WindowAttention preserves token sequence shape."""
        ws = 2
        N = ws ** 2
        nW, B = 4, 2
        dim = 16
        attn = WindowAttention(dim, ws, num_heads=2, is3d=False)
        x = torch.randn(nW * B, N, dim)
        y = attn(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# SwinTransformerBlock
# ---------------------------------------------------------------------------

class TestSwinTransformerBlock:
    def test_3d_unshifted_output_shape(self):
        """3D SwinTransformerBlock (shift_size=0) preserves input shape."""
        blk = SwinTransformerBlock(dim=16, num_heads=2, window_size=2,
                                   shift_size=0, is3d=True)
        x = torch.randn(2, 16, 8, 8, 8)
        assert blk(x).shape == x.shape

    def test_3d_shifted_output_shape(self):
        """3D SwinTransformerBlock (shift_size>0) preserves input shape."""
        blk = SwinTransformerBlock(dim=16, num_heads=2, window_size=2,
                                   shift_size=1, is3d=True)
        x = torch.randn(2, 16, 8, 8, 8)
        assert blk(x).shape == x.shape

    def test_2d_unshifted_output_shape(self):
        """2D SwinTransformerBlock (shift_size=0) preserves input shape."""
        blk = SwinTransformerBlock(dim=16, num_heads=2, window_size=2,
                                   shift_size=0, is3d=False)
        x = torch.randn(2, 16, 8, 8)
        assert blk(x).shape == x.shape

    def test_2d_shifted_output_shape(self):
        """2D SwinTransformerBlock (shift_size>0) preserves input shape."""
        blk = SwinTransformerBlock(dim=16, num_heads=2, window_size=2,
                                   shift_size=1, is3d=False)
        x = torch.randn(2, 16, 8, 8)
        assert blk(x).shape == x.shape

    def test_attn_mask_cached(self):
        """Attention mask is cached and reused on repeated calls with the same shape."""
        blk = SwinTransformerBlock(dim=16, num_heads=2, window_size=2,
                                   shift_size=1, is3d=True)
        x = torch.randn(2, 16, 8, 8, 8)
        blk(x)
        mask_first = blk._attn_mask
        blk(x)
        assert blk._attn_mask is mask_first, "Mask should be reused for the same shape"


# ---------------------------------------------------------------------------
# Package export
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_classes_importable_from_package(self):
        """All three Swin classes are importable from magnet_pinn.models."""
        from magnet_pinn.models import (
            SwinTransformerUNet,
            SwinTransformerUNet3D,
            SwinTransformerUNet2D,
        )
        assert SwinTransformerUNet3D is not None
        assert SwinTransformerUNet2D is not None

    def test_3d_is_subclass_of_base(self):
        """SwinTransformerUNet3D must be a subclass of SwinTransformerUNet."""
        from magnet_pinn.models import SwinTransformerUNet, SwinTransformerUNet3D
        assert issubclass(SwinTransformerUNet3D, SwinTransformerUNet)

    def test_2d_is_subclass_of_base(self):
        """SwinTransformerUNet2D must be a subclass of SwinTransformerUNet."""
        from magnet_pinn.models import SwinTransformerUNet, SwinTransformerUNet2D
        assert issubclass(SwinTransformerUNet2D, SwinTransformerUNet)
