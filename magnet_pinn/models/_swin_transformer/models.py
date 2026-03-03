"""Swin Transformer U-Net model implementations for 2D and 3D field prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buildingblocks import (
    PatchEmbedding,
    SwinStage,
    SwinDecoderStage,
)


class SwinTransformerUNet(nn.Module):
    """
    Swin Transformer U-Net for volumetric EM field prediction.

    Implements a hierarchical encoder-decoder architecture using Swin
    Transformer blocks in the encoder and convolutional refinement blocks
    in the decoder, connected by skip connections (Swin-UNETR style).

    The encoder produces features at ``len(depths)`` resolutions with
    progressively halved spatial dimensions and doubled channel widths.
    The last encoder stage acts as a bottleneck without spatial downscaling.
    The decoder reverses the resolution using PatchExpanding upsampling
    and merges encoder skip connections via concatenation.

    Arbitrary input spatial sizes are supported: the input is zero-padded
    to the nearest multiple of ``patch_size * window_size`` before the
    encoder, and the output is cropped back to the original spatial size.

    The forward interface is identical to the U-Net models in this package:
    ``forward(x)`` accepts ``(B, in_channels, D, H, W)`` and returns a
    tensor of shape ``(B, out_channels, D, H, W)`` with identical spatial
    dimensions.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        img_size (int or tuple): expected input spatial size; used to validate
            that the token grid is compatible with the window size
        patch_size (int): spatial stride of the patch embedding convolution;
            applied identically in all spatial dimensions
        embed_dim (int): base embedding dimension; doubled at each encoder stage
        depths (list of int): number of Swin Transformer blocks per encoder stage
        num_heads (list of int): number of attention heads per encoder stage;
            must have the same length as depths
        window_size (int): side length of the local attention window in tokens
        mlp_ratio (float): expansion ratio for the MLP hidden dimension
        qkv_bias (bool): add learnable bias to Q, K, and V projections
        dropout_prob (float): dropout probability for MLP and conv layers
        attn_dropout (float): dropout probability applied to attention weights
        is3d (bool): operate in 3D volumetric mode (True) or 2D planar mode (False)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        img_size=96,
        patch_size=4,
        embed_dim=96,
        depths=None,
        num_heads=None,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout_prob=0.1,
        attn_dropout=0.0,
        is3d=True,
    ):
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        assert len(depths) == len(num_heads), (
            f"depths and num_heads must have the same length, "
            f"got {len(depths)} and {len(num_heads)}"
        )

        self.is3d = is3d
        self.patch_size = patch_size
        self.window_size = window_size
        self.num_stages = len(depths)

        # --- Patch embedding ---
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            is3d=is3d,
        )

        # --- Encoder stages ---
        # Stage k operates at embed_dim * 2^k channels.
        # All stages except the last apply PatchMerging (downsample=True).
        # The last stage is the bottleneck: no spatial downscaling.
        self.encoder_stages = nn.ModuleList()
        for k in range(self.num_stages):
            stage = SwinStage(
                dim=embed_dim * (2 ** k),
                depth=depths[k],
                num_heads=num_heads[k],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout_prob=dropout_prob,
                attn_dropout=attn_dropout,
                downsample=(k < self.num_stages - 1),
                is3d=is3d,
            )
            self.encoder_stages.append(stage)

        # --- Decoder stages ---
        # Decoder stage i takes the output of the previous (deeper) level with
        # dim = embed_dim * 2^(num_stages-1-i) and merges with the skip from
        # encoder stage (num_stages-2-i) with dim = embed_dim * 2^(num_stages-2-i).
        # Output channels equal the skip channel count.
        #
        # Example for 4-stage net with embed_dim=E:
        #   dec[0]: (8E bottleneck) → expand(8E→4E) + concat(skip 4E) → refine → 4E
        #   dec[1]: (4E)            → expand(4E→2E) + concat(skip 2E) → refine → 2E
        #   dec[2]: (2E)            → expand(2E→E)  + concat(skip E)  → refine → E
        self.decoder_stages = nn.ModuleList()
        for k in range(self.num_stages - 1, 0, -1):
            decoder_in_dim = embed_dim * (2 ** k)
            skip_dim = embed_dim * (2 ** (k - 1))
            out_dim = skip_dim
            self.decoder_stages.append(
                SwinDecoderStage(
                    dim=decoder_in_dim,
                    skip_dim=skip_dim,
                    out_dim=out_dim,
                    is3d=is3d,
                    dropout_prob=dropout_prob,
                )
            )

        # --- Final upsample to recover patch_size factor ---
        # After the decoder, spatial resolution is input // patch_size.
        # Upsample by patch_size to recover the original (padded) resolution.
        upsample_mode = 'trilinear' if is3d else 'bilinear'
        self.final_upsample = nn.Upsample(
            scale_factor=patch_size,
            mode=upsample_mode,
            align_corners=False,
        )

        # --- Final 1x1 conv head ---
        Conv = nn.Conv3d if is3d else nn.Conv2d
        self.final_conv = Conv(embed_dim, out_channels, kernel_size=1)

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

    def _pad_input(self, x):
        """
        Zero-pad the input so every spatial dimension is divisible by
        ``patch_size * window_size * 2^(num_stages-1)``.

        This ensures both that:
        - the token grid is divisible by window_size (window attention tiling), and
        - the token grid can be halved (num_stages - 1) times by PatchMerging
          without producing odd spatial dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D, H, W) or (B, C, H, W).

        Returns
        -------
        x : torch.Tensor
            Padded tensor.
        orig_shape : tuple of int
            Original (unpadded) spatial dimensions, used for cropping the output.
        """
        # Token grid must be divisible by window_size and by 2^(num_stages-1)
        divisor = self.patch_size * self.window_size * (2 ** (self.num_stages - 1))

        if self.is3d:
            B, C, D, H, W = x.shape
            pad_d = (divisor - D % divisor) % divisor
            pad_h = (divisor - H % divisor) % divisor
            pad_w = (divisor - W % divisor) % divisor
            # F.pad expects padding in reverse dim order: (last, ..., first)
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            return x, (D, H, W)
        else:
            B, C, H, W = x.shape
            pad_h = (divisor - H % divisor) % divisor
            pad_w = (divisor - W % divisor) % divisor
            x = F.pad(x, (0, pad_w, 0, pad_h))
            return x, (H, W)

    def _crop_output(self, x, orig_shape):
        """
        Crop the output tensor to the original input spatial dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Output tensor after upsampling, potentially with padded boundary.
        orig_shape : tuple of int
            Original spatial dimensions returned by _pad_input.

        Returns
        -------
        torch.Tensor
            Cropped tensor matching the original spatial shape.
        """
        if self.is3d:
            D, H, W = orig_shape
            return x[:, :, :D, :H, :W]
        else:
            H, W = orig_shape
            return x[:, :, :H, :W]

    def forward(self, x):
        """
        Forward pass through the Swin Transformer U-Net.

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
        # 1. Pad to multiple of patch_size * window_size
        x, orig_shape = self._pad_input(x)

        # 2. Patch embedding: (B, C, ...) → (B, embed_dim, D//ps, H//ps, W//ps)
        x = self.patch_embed(x)

        # 3. Encoder: collect skip connections before PatchMerging
        #    We iterate stage.blocks and stage.downsample separately so the
        #    pre-merge feature map is available as the skip connection.
        skips = []
        for stage in self.encoder_stages:
            for blk in stage.blocks:
                x = blk(x)
            skips.append(x)                     # save pre-merge feature map
            if stage.downsample is not None:
                x = stage.downsample(x)

        # skips[-1]  = bottleneck features (same as current x, no downsampling)
        # skips[:-1] = encoder skip connections, from coarsest to finest

        # 4. Decoder: x starts as the bottleneck output
        #    skip_connections is [skip_{N-2}, skip_{N-3}, ..., skip_0]
        skip_connections = list(reversed(skips[:-1]))
        for dec_stage, skip in zip(self.decoder_stages, skip_connections):
            x = dec_stage(x, skip)

        # 5. Upsample by patch_size to recover original token-grid resolution
        x = self.final_upsample(x)

        # 6. 1×1 conv head: embed_dim → out_channels
        x = self.final_conv(x)

        # 7. Crop to original spatial dimensions
        x = self._crop_output(x, orig_shape)

        return x


class SwinTransformerUNet3D(SwinTransformerUNet):
    """
    3D Swin Transformer U-Net with is3d=True pre-configured.

    Uses Swin-Tiny default hyperparameters scaled for 3D volumetric data.
    The default window size of 7 gives a 7×7×7 local attention window.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        img_size (int or tuple): expected input spatial size (default: 96)
        patch_size (int): patch embedding stride (default: 4)
        embed_dim (int): base embedding dimension (default: 96)
        depths (list of int): Swin blocks per encoder stage (default: [2, 2, 6, 2])
        num_heads (list of int): attention heads per stage (default: [3, 6, 12, 24])
        window_size (int): attention window side length in tokens (default: 7)
        mlp_ratio (float): MLP hidden dimension expansion factor (default: 4.0)
        qkv_bias (bool): learnable Q/K/V bias (default: True)
        dropout_prob (float): general dropout probability (default: 0.1)
        attn_dropout (float): attention weight dropout (default: 0.0)
        **kwargs: additional keyword arguments forwarded to SwinTransformerUNet
    """

    def __init__(self, in_channels, out_channels, img_size=96, patch_size=4,
                 embed_dim=96, depths=None, num_heads=None, window_size=7,
                 mlp_ratio=4.0, qkv_bias=True, dropout_prob=0.1,
                 attn_dropout=0.0, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout_prob=dropout_prob,
            attn_dropout=attn_dropout,
            is3d=True,
        )


class SwinTransformerUNet2D(SwinTransformerUNet):
    """
    2D Swin Transformer U-Net with is3d=False pre-configured.

    Uses the same Swin-Tiny defaults as SwinTransformerUNet3D but operates
    on 2D planar inputs. img_size can be a scalar or a (H, W) tuple.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        img_size (int or tuple): expected input spatial size (default: 96)
        patch_size (int): patch embedding stride (default: 4)
        embed_dim (int): base embedding dimension (default: 96)
        depths (list of int): Swin blocks per encoder stage (default: [2, 2, 6, 2])
        num_heads (list of int): attention heads per stage (default: [3, 6, 12, 24])
        window_size (int): attention window side length in tokens (default: 7)
        mlp_ratio (float): MLP hidden dimension expansion factor (default: 4.0)
        qkv_bias (bool): learnable Q/K/V bias (default: True)
        dropout_prob (float): general dropout probability (default: 0.1)
        attn_dropout (float): attention weight dropout (default: 0.0)
        **kwargs: additional keyword arguments forwarded to SwinTransformerUNet
    """

    def __init__(self, in_channels, out_channels, img_size=96, patch_size=4,
                 embed_dim=96, depths=None, num_heads=None, window_size=7,
                 mlp_ratio=4.0, qkv_bias=True, dropout_prob=0.1,
                 attn_dropout=0.0, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout_prob=dropout_prob,
            attn_dropout=attn_dropout,
            is3d=False,
        )
