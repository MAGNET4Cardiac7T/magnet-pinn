"""Tests for the Transolver (Physics-Attention Transformer) implementation."""

import pytest
import torch

from magnet_pinn.models._transsolver.buildingblocks import (
    PhysicsAttention,
    TransolverBlock,
)


# ---------------------------------------------------------------------------
# Output shape — 3D
# ---------------------------------------------------------------------------

class TestOutputShape3D:
    def test_aligned_input(self, batch_size, in_channels, out_channels,
                           spatial_size_3d, transolver3d_model):
        """Model returns same spatial dims as input."""
        s = spatial_size_3d
        x = torch.randn(batch_size, in_channels, s, s, s)
        with torch.no_grad():
            y = transolver3d_model(x)
        assert y.shape == (batch_size, out_channels, s, s, s)

    def test_arbitrary_input(self, batch_size, in_channels, out_channels,
                             transolver3d_model):
        """Model handles arbitrary (non-aligned) spatial sizes without padding."""
        x = torch.randn(batch_size, in_channels, 13, 15, 11)
        with torch.no_grad():
            y = transolver3d_model(x)
        assert y.shape == (batch_size, out_channels, 13, 15, 11)

    def test_non_cubic_input(self, batch_size, in_channels, out_channels,
                             transolver3d_model):
        """Model handles non-cubic inputs with different extents per dimension."""
        x = torch.randn(batch_size, in_channels, 8, 16, 24)
        with torch.no_grad():
            y = transolver3d_model(x)
        assert y.shape == (batch_size, out_channels, 8, 16, 24)

    def test_single_batch(self, in_channels, out_channels, transolver3d_model):
        """Model works with batch_size=1."""
        x = torch.randn(1, in_channels, 16, 16, 16)
        with torch.no_grad():
            y = transolver3d_model(x)
        assert y.shape == (1, out_channels, 16, 16, 16)


# ---------------------------------------------------------------------------
# Output shape — 2D
# ---------------------------------------------------------------------------

class TestOutputShape2D:
    def test_aligned_input(self, batch_size, in_channels, out_channels,
                           spatial_size_2d, transolver2d_model):
        """2D model returns same spatial dims for aligned input."""
        s = spatial_size_2d
        x = torch.randn(batch_size, in_channels, s, s)
        with torch.no_grad():
            y = transolver2d_model(x)
        assert y.shape == (batch_size, out_channels, s, s)

    def test_arbitrary_input(self, batch_size, in_channels, out_channels,
                             transolver2d_model):
        """2D model handles arbitrary spatial sizes."""
        x = torch.randn(batch_size, in_channels, 13, 17)
        with torch.no_grad():
            y = transolver2d_model(x)
        assert y.shape == (batch_size, out_channels, 13, 17)

    def test_non_square_input(self, batch_size, in_channels, out_channels,
                              transolver2d_model):
        """2D model handles non-square inputs."""
        x = torch.randn(batch_size, in_channels, 8, 16)
        with torch.no_grad():
            y = transolver2d_model(x)
        assert y.shape == (batch_size, out_channels, 8, 16)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_3d_gradients_not_nan(self, batch_size, in_channels,
                                  transolver3d_model):
        """All parameters receive non-NaN gradients from a scalar loss."""
        x = torch.randn(batch_size, in_channels, 8, 8, 8)
        transolver3d_model.train()
        y = transolver3d_model(x)
        y.mean().backward()
        for name, p in transolver3d_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for parameter: {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient for parameter: {name}"

    def test_2d_gradients_not_nan(self, batch_size, in_channels,
                                  transolver2d_model):
        """2D model: all parameters receive non-NaN gradients."""
        x = torch.randn(batch_size, in_channels, 8, 8)
        transolver2d_model.train()
        y = transolver2d_model(x)
        y.mean().backward()
        for name, p in transolver2d_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for parameter: {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient for parameter: {name}"


# ---------------------------------------------------------------------------
# Determinism in eval mode
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_3d_deterministic_in_eval(self, batch_size, in_channels,
                                      transolver3d_model):
        """Two consecutive eval-mode forward passes produce identical outputs."""
        x = torch.randn(batch_size, in_channels, 8, 8, 8)
        transolver3d_model.eval()
        with torch.no_grad():
            y1 = transolver3d_model(x)
            y2 = transolver3d_model(x)
        assert torch.allclose(y1, y2)

    def test_2d_deterministic_in_eval(self, batch_size, in_channels,
                                      transolver2d_model):
        """2D model: two eval-mode forward passes are identical."""
        x = torch.randn(batch_size, in_channels, 8, 8)
        transolver2d_model.eval()
        with torch.no_grad():
            y1 = transolver2d_model(x)
            y2 = transolver2d_model(x)
        assert torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# PhysicsAttention unit tests
# ---------------------------------------------------------------------------

class TestPhysicsAttention:
    def test_output_shape(self):
        """PhysicsAttention preserves sequence shape (B, N, C)."""
        B, N, C = 2, 64, 16
        attn = PhysicsAttention(embed_dim=C, n_slices=8, num_heads=2)
        x = torch.randn(B, N, C)
        y = attn(x)
        assert y.shape == (B, N, C)

    def test_slice_weights_sum_to_one(self):
        """Slice weights must sum to 1 over the slice dimension (softmax invariant)."""
        B, N, C = 2, 32, 16
        n_slices = 8
        attn = PhysicsAttention(embed_dim=C, n_slices=n_slices, num_heads=2)
        x = torch.randn(B, N, C)
        M = torch.softmax(attn.slice_proj(x), dim=-1)  # (B, N, n_slices)
        sums = M.sum(dim=-1)  # (B, N)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_different_n_values(self):
        """PhysicsAttention works for any N (key advantage over window attention)."""
        attn = PhysicsAttention(embed_dim=16, n_slices=4, num_heads=2)
        for N in [10, 100, 1000]:
            x = torch.randn(1, N, 16)
            y = attn(x)
            assert y.shape == (1, N, 16), f"Failed for N={N}"

    def test_embed_dim_not_divisible_raises(self):
        """PhysicsAttention must raise AssertionError when embed_dim % num_heads != 0."""
        with pytest.raises(AssertionError):
            PhysicsAttention(embed_dim=10, n_slices=4, num_heads=3)


# ---------------------------------------------------------------------------
# TransolverBlock unit tests
# ---------------------------------------------------------------------------

class TestTransolverBlock:
    def test_output_shape(self):
        """TransolverBlock preserves (B, N, C) shape."""
        B, N, C = 2, 64, 16
        blk = TransolverBlock(embed_dim=C, n_slices=8, num_heads=2)
        x = torch.randn(B, N, C)
        y = blk(x)
        assert y.shape == (B, N, C)

    def test_residual_connection(self):
        """With zero-initialized weights, block approximates identity via residual."""
        B, N, C = 2, 16, 8
        blk = TransolverBlock(embed_dim=C, n_slices=4, num_heads=2)
        with torch.no_grad():
            for p in blk.parameters():
                p.zero_()
        x = torch.randn(B, N, C)
        y = blk(x)
        # All-zero weights: LN(zeros)=zeros, attn output=zeros, mlp output=zeros
        # → residual path returns x + 0 = x
        assert torch.allclose(y, x, atol=1e-5)

    def test_arbitrary_sequence_length(self):
        """Block works for any N without spatial constraints."""
        blk = TransolverBlock(embed_dim=8, n_slices=4, num_heads=2)
        for N in [7, 100, 512]:
            x = torch.randn(1, N, 8)
            assert blk(x).shape == (1, N, 8)


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_classes_importable_from_package(self):
        """All three Transolver classes are importable from magnet_pinn.models."""
        from magnet_pinn.models import Transolver, Transolver3D, Transolver2D
        assert Transolver is not None
        assert Transolver3D is not None
        assert Transolver2D is not None

    def test_3d_is_subclass_of_base(self):
        """Transolver3D must be a subclass of Transolver."""
        from magnet_pinn.models import Transolver, Transolver3D
        assert issubclass(Transolver3D, Transolver)

    def test_2d_is_subclass_of_base(self):
        """Transolver2D must be a subclass of Transolver."""
        from magnet_pinn.models import Transolver, Transolver2D
        assert issubclass(Transolver2D, Transolver)

    def test_in_all_list(self):
        """All exported names appear in magnet_pinn.models.__all__."""
        import magnet_pinn.models as m
        for name in ["Transolver", "Transolver3D", "Transolver2D"]:
            assert name in m.__all__, f"{name} missing from __all__"
