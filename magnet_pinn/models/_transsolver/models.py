"""Transolver model implementations for 2D and 3D field prediction."""

import torch
import torch.nn as nn

from .buildingblocks import TransolverBlock


class Transolver(nn.Module):
    """
    Transolver: Physics-Attention Transformer for PDE field prediction.

    Implements the architecture from "Transolver: A Fast Transformer Solver
    for PDEs on General Geometries" (Wu et al., 2024).

    The model accepts standard grid tensors (B, C, spatial...) and internally
    reshapes them to sequences (B, N, C), applies a stack of TransolverBlocks
    with Physics-Attention, then reshapes the output back to the original
    spatial layout. Arbitrary spatial sizes are supported with no padding
    requirement (unlike window-based models).

    The core innovation is Physics-Attention: instead of O(N^2) attention over
    all mesh points, each point is softly assigned to one of n_slices physics-
    state tokens. Attention runs only among the n_slices tokens (O(n_slices^2)),
    then results are scattered back to each point via the learned assignments.

    The forward interface is identical to the other models in this package:
    ``forward(x)`` accepts ``(B, in_channels, D, H, W)`` and returns a
    tensor of shape ``(B, out_channels, D, H, W)`` with identical spatial
    dimensions.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        embed_dim (int): internal embedding dimension (default: 64)
        depth (int): number of stacked TransolverBlocks (default: 6)
        n_slices (int): number of physics-state slices in PhysicsAttention (default: 64)
        num_heads (int): attention heads for slice-level attention (default: 8)
        mlp_ratio (float): MLP hidden dimension expansion factor (default: 4.0)
        qkv_bias (bool): learnable bias in attention (default: True)
        dropout_prob (float): MLP and projection dropout (default: 0.0)
        attn_dropout (float): attention weight dropout (default: 0.0)
        pos_encoding (bool): add sinusoidal position encoding from grid
            coordinates (default: False); the physics-aware slice mechanism
            implicitly encodes spatial structure, so this is optional
        is3d (bool): 3D volumetric input (True) or 2D planar input (False)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int = 64,
        depth: int = 6,
        n_slices: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout_prob: float = 0.0,
        attn_dropout: float = 0.0,
        pos_encoding: bool = False,
        is3d: bool = True,
    ):
        super().__init__()
        self.is3d = is3d
        self.pos_encoding = pos_encoding

        # Input projection: in_channels → embed_dim (applied per-point)
        self.input_proj = nn.Linear(in_channels, embed_dim)

        # Optional sinusoidal position encoding: normalized grid coords → embed_dim
        # Coords are generated on-the-fly, supporting arbitrary spatial sizes.
        if pos_encoding:
            coord_dim = 3 if is3d else 2
            self.pos_proj = nn.Linear(coord_dim, embed_dim, bias=False)
        else:
            self.pos_proj = None

        # Stack of TransolverBlocks
        self.blocks = nn.ModuleList([
            TransolverBlock(
                embed_dim=embed_dim,
                n_slices=n_slices,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout_prob=dropout_prob,
                attn_dropout=attn_dropout,
            )
            for _ in range(depth)
        ])

        # Final norm before output head (ViT-style)
        self.norm = nn.LayerNorm(embed_dim)

        # Output head: embed_dim → out_channels
        self.output_proj = nn.Linear(embed_dim, out_channels)

        self._init_weights()

    def _init_weights(self):
        """Apply truncated-normal init to Linear layers and standard init to LayerNorm."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _grid_coords(self, spatial_shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Compute normalized voxel grid coordinates in [0, 1]^d, generated on-the-fly.

        Parameters
        ----------
        spatial_shape : tuple of int
            (D, H, W) for 3D or (H, W) for 2D.
        device : torch.device
            Target device.

        Returns
        -------
        torch.Tensor
            Shape (1, N, d) where N = prod(spatial_shape), d = len(spatial_shape).
        """
        grids = [torch.linspace(0, 1, s, device=device) for s in spatial_shape]
        coords = torch.stack(torch.meshgrid(*grids, indexing='ij'), dim=-1)
        N = coords[..., 0].numel()
        return coords.view(N, len(spatial_shape)).unsqueeze(0)  # (1, N, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transolver.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, D, H, W) for is3d=True
            or (B, in_channels, H, W) for is3d=False.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, D, H, W) or
            (B, out_channels, H, W), with the same spatial dimensions as
            the input.
        """
        # 1. Record spatial shape and reshape grid → sequence
        if self.is3d:
            B, C, D, H, W = x.shape
            spatial_shape = (D, H, W)
            x = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)
        else:
            B, C, H, W = x.shape
            spatial_shape = (H, W)
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # 2. Input projection: in_channels → embed_dim
        x = self.input_proj(x)  # (B, N, embed_dim)

        # 3. Optional position encoding: add normalized grid coordinates
        if self.pos_proj is not None:
            coords = self._grid_coords(spatial_shape, x.device)  # (1, N, d)
            x = x + self.pos_proj(coords)  # broadcast over batch

        # 4. Transolver blocks
        for blk in self.blocks:
            x = blk(x)

        # 5. Final norm + output projection
        x = self.norm(x)
        x = self.output_proj(x)  # (B, N, out_channels)

        # 6. Reshape sequence → grid
        if self.is3d:
            x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
        else:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return x


class Transolver3D(Transolver):
    """
    3D Transolver with is3d=True pre-configured.

    Processes volumetric inputs of shape (B, in_channels, D, H, W) and
    returns outputs of shape (B, out_channels, D, H, W).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        embed_dim (int): internal embedding dimension (default: 64)
        depth (int): number of TransolverBlocks (default: 6)
        n_slices (int): physics-state slices in PhysicsAttention (default: 64)
        num_heads (int): attention heads for slice attention (default: 8)
        mlp_ratio (float): MLP expansion ratio (default: 4.0)
        qkv_bias (bool): learnable QKV bias (default: True)
        dropout_prob (float): general dropout probability (default: 0.0)
        attn_dropout (float): attention weight dropout (default: 0.0)
        pos_encoding (bool): sinusoidal position encoding (default: False)
        **kwargs: additional keyword arguments forwarded to Transolver
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int = 64,
        depth: int = 6,
        n_slices: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout_prob: float = 0.0,
        attn_dropout: float = 0.0,
        pos_encoding: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            n_slices=n_slices,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout_prob=dropout_prob,
            attn_dropout=attn_dropout,
            pos_encoding=pos_encoding,
            is3d=True,
        )


class Transolver2D(Transolver):
    """
    2D Transolver with is3d=False pre-configured.

    Processes planar inputs of shape (B, in_channels, H, W) and
    returns outputs of shape (B, out_channels, H, W).

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        embed_dim (int): internal embedding dimension (default: 64)
        depth (int): number of TransolverBlocks (default: 6)
        n_slices (int): physics-state slices in PhysicsAttention (default: 64)
        num_heads (int): attention heads for slice attention (default: 8)
        mlp_ratio (float): MLP expansion ratio (default: 4.0)
        qkv_bias (bool): learnable QKV bias (default: True)
        dropout_prob (float): general dropout probability (default: 0.0)
        attn_dropout (float): attention weight dropout (default: 0.0)
        pos_encoding (bool): sinusoidal position encoding (default: False)
        **kwargs: additional keyword arguments forwarded to Transolver
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int = 64,
        depth: int = 6,
        n_slices: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout_prob: float = 0.0,
        attn_dropout: float = 0.0,
        pos_encoding: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            n_slices=n_slices,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout_prob=dropout_prob,
            attn_dropout=attn_dropout,
            pos_encoding=pos_encoding,
            is3d=False,
        )
