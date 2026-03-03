"""Building blocks for Swin Transformer U-Net architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _safe_num_groups(channels, max_groups=8):
    """
    Return the largest power-of-2 that divides channels and is <= max_groups.

    Args:
        channels (int): number of channels
        max_groups (int): maximum number of groups

    Returns:
        int: number of groups for GroupNorm
    """
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g //= 2
    return g


def window_partition(x, window_size):
    """
    Partition a 3D feature map into non-overlapping windows.

    Args:
        x (torch.Tensor): input tensor of shape (B, D, H, W, C)
        window_size (int): side length of each cubic window

    Returns:
        torch.Tensor: windows of shape (num_windows * B, window_size, window_size, window_size, C)
    """
    B, D, H, W, C = x.shape
    ws = window_size
    x = x.view(B, D // ws, ws, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, ws, ws, ws, C)
    return windows


def window_reverse(windows, window_size, D, H, W):
    """
    Reconstruct a 3D feature map from non-overlapping windows.

    Args:
        windows (torch.Tensor): tensor of shape (num_windows * B, ws, ws, ws, C)
        window_size (int): side length of each cubic window
        D (int): depth of the original feature map
        H (int): height of the original feature map
        W (int): width of the original feature map

    Returns:
        torch.Tensor: reconstructed tensor of shape (B, D, H, W, C)
    """
    ws = window_size
    B = int(windows.shape[0] / (D * H * W / ws ** 3))
    x = windows.view(B, D // ws, H // ws, W // ws, ws, ws, ws, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    return x


def window_partition_2d(x, window_size):
    """
    Partition a 2D feature map into non-overlapping windows.

    Args:
        x (torch.Tensor): input tensor of shape (B, H, W, C)
        window_size (int): side length of each square window

    Returns:
        torch.Tensor: windows of shape (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    ws = window_size
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, ws, ws, C)
    return windows


def window_reverse_2d(windows, window_size, H, W):
    """
    Reconstruct a 2D feature map from non-overlapping windows.

    Args:
        windows (torch.Tensor): tensor of shape (num_windows * B, ws, ws, C)
        window_size (int): side length of each square window
        H (int): height of the original feature map
        W (int): width of the original feature map

    Returns:
        torch.Tensor: reconstructed tensor of shape (B, H, W, C)
    """
    ws = window_size
    B = int(windows.shape[0] / (H * W / ws ** 2))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class PatchEmbedding(nn.Module):
    """
    Converts a volumetric or planar image into non-overlapping patch tokens.

    Uses a strided convolution as the embedding: the convolution stride equals
    patch_size, giving exactly one token per patch. A LayerNorm is applied
    to the resulting token embeddings.

    Args:
        in_channels (int): number of input image channels
        embed_dim (int): output token embedding dimension
        patch_size (int): spatial stride of the embedding convolution; identical
            stride is applied in all spatial dimensions
        is3d (bool): use Conv3d (True) or Conv2d (False)
    """

    def __init__(self, in_channels, embed_dim, patch_size=4, is3d=True):
        super().__init__()
        self.is3d = is3d
        if is3d:
            self.proj = nn.Conv3d(in_channels, embed_dim,
                                  kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_channels, embed_dim,
                                  kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D, H, W) for is3d=True or
            (B, C, H, W) for is3d=False.

        Returns
        -------
        torch.Tensor
            Patch token feature map of shape (B, embed_dim, D//ps, H//ps, W//ps)
            or (B, embed_dim, H//ps, W//ps).
        """
        x = self.proj(x)
        # Permute to channel-last for LayerNorm, then restore
        if self.is3d:
            B, C, D, H, W = x.shape
            x = x.permute(0, 2, 3, 4, 1)   # (B, D, H, W, C)
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3)   # (B, C, D, H, W)
        else:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)       # (B, H, W, C)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)       # (B, C, H, W)
        return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.

    Implements W-MSA and SW-MSA from "Swin Transformer: Hierarchical Vision
    Transformer using Shifted Windows" (Liu et al., 2021). The relative
    position bias is stored in a learnable table and indexed via a pre-computed
    index buffer.

    Args:
        dim (int): token embedding dimension
        window_size (int): side length of the cubic or square attention window
        num_heads (int): number of attention heads
        qkv_bias (bool): add learnable bias to Q, K, V projections
        attn_dropout (float): dropout applied to attention weights
        proj_dropout (float): dropout applied after the output projection
        is3d (bool): 3D windows (True) or 2D windows (False)
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_dropout=0.0, proj_dropout=0.0, is3d=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.is3d = is3d
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        if is3d:
            table_size = (2 * window_size - 1) ** 3
        else:
            table_size = (2 * window_size - 1) ** 2

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        relative_position_index = self._compute_relative_position_index(
            window_size, is3d
        )
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    @staticmethod
    def _compute_relative_position_index(window_size, is3d):
        """
        Pre-compute an integer index tensor mapping each token pair to a row
        in the relative position bias table.

        Parameters
        ----------
        window_size : int
            Side length of the attention window.
        is3d : bool
            3D (True) or 2D (False) windows.

        Returns
        -------
        torch.Tensor
            Integer index tensor of shape (N, N) where N = ws^3 (3D) or ws^2 (2D).
        """
        ws = window_size
        if is3d:
            coords_d = torch.arange(ws)
            coords_h = torch.arange(ws)
            coords_w = torch.arange(ws)
            coords = torch.stack(
                torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij')
            )  # (3, ws, ws, ws)
            coords_flat = coords.flatten(1)  # (3, ws^3)

            relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (3, N, N)
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()      # (N, N, 3)
            relative_coords[:, :, 0] += ws - 1
            relative_coords[:, :, 1] += ws - 1
            relative_coords[:, :, 2] += ws - 1
            relative_coords[:, :, 0] *= (2 * ws - 1) ** 2
            relative_coords[:, :, 1] *= (2 * ws - 1)
        else:
            coords_h = torch.arange(ws)
            coords_w = torch.arange(ws)
            coords = torch.stack(
                torch.meshgrid(coords_h, coords_w, indexing='ij')
            )  # (2, ws, ws)
            coords_flat = coords.flatten(1)  # (2, ws^2)

            relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()      # (N, N, 2)
            relative_coords[:, :, 0] += ws - 1
            relative_coords[:, :, 1] += ws - 1
            relative_coords[:, :, 0] *= (2 * ws - 1)

        return relative_coords.sum(-1)  # (N, N)

    def forward(self, x, mask=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Token sequence of shape (num_windows * B, N, C) where N = ws^3 (3D)
            or ws^2 (2D).
        mask : torch.Tensor or None
            Attention mask for shifted-window attention, shape (num_windows, N, N).
            Positions with large negative values (-100) are masked out.

        Returns
        -------
        torch.Tensor
            Output of same shape as input.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias: (num_heads, N, N)
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, self.num_heads)
        bias = bias.permute(2, 0, 1)  # (num_heads, N, N)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class SwinTransformerBlock(nn.Module):
    """
    A single Swin Transformer block with window (or shifted-window) attention
    and an MLP sub-layer.

    Each block applies:
        x = x + W-MSA(LN(x))   [shift_size=0: plain windows]
        x = x + MLP(LN(x))
    or with cyclic shift (shift_size > 0):
        x = x + SW-MSA(LN(x))

    The attention mask for shifted windows is computed lazily on the first
    forward pass and cached per spatial shape.

    Args:
        dim (int): token embedding dimension
        num_heads (int): number of attention heads
        window_size (int): attention window side length
        shift_size (int): cyclic shift amount; 0 disables shifting
        mlp_ratio (float): MLP hidden dimension expansion factor
        qkv_bias (bool): learnable Q/K/V bias
        dropout_prob (float): MLP and projection dropout probability
        attn_dropout (float): attention weight dropout probability
        is3d (bool): 3D or 2D mode
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True,
                 dropout_prob=0.0, attn_dropout=0.0, is3d=True):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.window_size = window_size
        self.is3d = is3d

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout_prob,
            is3d=is3d,
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout_prob),
        )

        # Attention mask cached per spatial shape; recomputed on shape change
        self._attn_mask = None
        self._attn_mask_shape = None

    def _compute_attn_mask(self, spatial_shape, device):
        """
        Compute the SW-MSA attention mask for a given spatial shape.

        Tokens that originate from different image regions (after cyclic shift)
        but land in the same window are masked by adding -100 to their
        pre-softmax attention logits.

        Parameters
        ----------
        spatial_shape : tuple of int
            (D, H, W) for 3D or (H, W) for 2D.
        device : torch.device
            Device on which to create the mask tensor.

        Returns
        -------
        torch.Tensor or None
            Mask tensor of shape (num_windows, N, N), or None if shift_size=0.
        """
        if self.shift_size == 0:
            return None

        ws = self.window_size
        ss = self.shift_size

        if self.is3d:
            D, H, W = spatial_shape
            img_mask = torch.zeros(1, D, H, W, 1, device=device)
            d_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
            h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
            w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
            cnt = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, d, h, w, :] = cnt
                        cnt += 1
            mask_windows = window_partition(img_mask, ws)         # (nW, ws, ws, ws, 1)
            mask_windows = mask_windows.view(-1, ws * ws * ws)
        else:
            H, W = spatial_shape
            img_mask = torch.zeros(1, H, W, 1, device=device)
            h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
            w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition_2d(img_mask, ws)      # (nW, ws, ws, 1)
            mask_windows = mask_windows.view(-1, ws * ws)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature map of shape (B, C, D, H, W) for is3d=True or
            (B, C, H, W) for is3d=False. C must equal self.dim.

        Returns
        -------
        torch.Tensor
            Output feature map of the same shape as the input.
        """
        if self.is3d:
            B, C, D, H, W = x.shape
            spatial = (D, H, W)
        else:
            B, C, H, W = x.shape
            spatial = (H, W)

        # Recompute (and cache) the attention mask when spatial shape changes
        if self._attn_mask_shape != spatial:
            self._attn_mask = self._compute_attn_mask(spatial, x.device)
            self._attn_mask_shape = spatial
        elif self._attn_mask is not None and self._attn_mask.device != x.device:
            self._attn_mask = self._attn_mask.to(x.device)

        # Permute to channel-last for attention computation
        if self.is3d:
            x_perm = x.permute(0, 2, 3, 4, 1)   # (B, D, H, W, C)
        else:
            x_perm = x.permute(0, 2, 3, 1)        # (B, H, W, C)

        shortcut = x_perm
        x_perm = self.norm1(x_perm)

        # Cyclic shift
        if self.shift_size > 0:
            if self.is3d:
                x_perm = torch.roll(x_perm, shifts=(-self.shift_size,) * 3, dims=(1, 2, 3))
            else:
                x_perm = torch.roll(x_perm, shifts=(-self.shift_size,) * 2, dims=(1, 2))

        # Partition into windows and flatten token dim
        if self.is3d:
            x_windows = window_partition(x_perm, self.window_size)         # (nW*B, ws, ws, ws, C)
            x_windows = x_windows.view(-1, self.window_size ** 3, C)
        else:
            x_windows = window_partition_2d(x_perm, self.window_size)      # (nW*B, ws, ws, C)
            x_windows = x_windows.view(-1, self.window_size ** 2, C)

        # Window attention
        attn_windows = self.attn(x_windows, mask=self._attn_mask)

        # Reverse partition
        if self.is3d:
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
            x_perm = window_reverse(attn_windows, self.window_size, D, H, W)
        else:
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            x_perm = window_reverse_2d(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            if self.is3d:
                x_perm = torch.roll(x_perm, shifts=(self.shift_size,) * 3, dims=(1, 2, 3))
            else:
                x_perm = torch.roll(x_perm, shifts=(self.shift_size,) * 2, dims=(1, 2))

        # First residual connection
        x_perm = shortcut + x_perm

        # MLP with second residual connection
        x_perm = x_perm + self.mlp(self.norm2(x_perm))

        # Restore channel-first layout
        if self.is3d:
            return x_perm.permute(0, 4, 1, 2, 3)
        else:
            return x_perm.permute(0, 3, 1, 2)


class PatchMerging(nn.Module):
    """
    Halves the spatial resolution and doubles the channel count.

    Samples a stride-2 grid of neighbouring tokens (2×2×2 for 3D, 2×2 for 2D),
    concatenates them along the channel axis, normalises, and projects to
    2× the input channels. This is the downscaling operation in the Swin encoder.

    Args:
        dim (int): number of input channels
        is3d (bool): 3D (True) or 2D (False) mode
    """

    def __init__(self, dim, is3d=True):
        super().__init__()
        self.is3d = is3d
        factor = 8 if is3d else 4
        self.norm = nn.LayerNorm(factor * dim)
        self.reduction = nn.Linear(factor * dim, 2 * dim, bias=False)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature map of shape (B, C, D, H, W) or (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Downsampled feature map of shape (B, 2C, D//2, H//2, W//2) or
            (B, 2C, H//2, W//2).
        """
        if self.is3d:
            x = x.permute(0, 2, 3, 4, 1)   # (B, D, H, W, C)
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 1::2, 1::2, 0::2, :]
            x4 = x[:, 0::2, 0::2, 1::2, :]
            x5 = x[:, 1::2, 0::2, 1::2, :]
            x6 = x[:, 0::2, 1::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)   # (B, D//2, H//2, W//2, 8C)
            x = self.norm(x)
            x = self.reduction(x)                                         # (B, D//2, H//2, W//2, 2C)
            return x.permute(0, 4, 1, 2, 3)
        else:
            x = x.permute(0, 2, 3, 1)       # (B, H, W, C)
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], dim=-1)     # (B, H//2, W//2, 4C)
            x = self.norm(x)
            x = self.reduction(x)                         # (B, H//2, W//2, 2C)
            return x.permute(0, 3, 1, 2)


class PatchExpanding(nn.Module):
    """
    Doubles the spatial resolution and halves the channel count.

    Projects to an expanded channel count then uses einops to rearrange
    the extra channels into spatial positions (pixel-shuffle / depth-to-space).
    This is the upscaling operation used in the Swin decoder.

    Args:
        dim (int): number of input channels; must be even
        is3d (bool): 3D (True) or 2D (False) mode
    """

    def __init__(self, dim, is3d=True):
        assert dim % 2 == 0, f"PatchExpanding requires an even channel count, got {dim}"
        super().__init__()
        self.is3d = is3d
        # Project to expand_factor * (dim//2): rearrange gives 2x in each spatial dim
        expand_factor = 8 if is3d else 4
        self.expand = nn.Linear(dim, expand_factor * (dim // 2), bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature map of shape (B, C, D, H, W) or (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Upsampled feature map of shape (B, C//2, 2D, 2H, 2W) or
            (B, C//2, 2H, 2W).
        """
        if self.is3d:
            x = x.permute(0, 2, 3, 4, 1)    # (B, D, H, W, C)
            x = self.expand(x)               # (B, D, H, W, 8*(C//2))
            x = rearrange(x, 'b d h w (p1 p2 p3 c) -> b (d p1) (h p2) (w p3) c',
                          p1=2, p2=2, p3=2)
            x = self.norm(x)
            return x.permute(0, 4, 1, 2, 3)
        else:
            x = x.permute(0, 2, 3, 1)        # (B, H, W, C)
            x = self.expand(x)               # (B, H, W, 4*(C//2))
            x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2)
            x = self.norm(x)
            return x.permute(0, 3, 1, 2)


class SwinStage(nn.Module):
    """
    One encoder stage consisting of N alternating W-MSA / SW-MSA blocks
    followed by an optional PatchMerging downscale.

    Blocks at even indices use shift_size=0 (plain window attention); blocks
    at odd indices use shift_size=window_size//2 (shifted window attention).

    The stage exposes self.blocks and self.downsample separately so that the
    parent model can collect skip connections between them.

    Args:
        dim (int): embedding dimension for this stage
        depth (int): number of SwinTransformerBlocks
        num_heads (int): number of attention heads
        window_size (int): attention window side length
        mlp_ratio (float): MLP hidden dimension expansion ratio
        qkv_bias (bool): learnable Q/K/V bias
        dropout_prob (float): MLP and projection dropout probability
        attn_dropout (float): attention weight dropout probability
        downsample (bool): whether to apply PatchMerging after the blocks
        is3d (bool): 3D or 2D mode
    """

    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0,
                 qkv_bias=True, dropout_prob=0.0, attn_dropout=0.0,
                 downsample=True, is3d=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout_prob=dropout_prob,
                attn_dropout=attn_dropout,
                is3d=is3d,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim, is3d=is3d) if downsample else None

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature map of shape (B, C, D, H, W) or (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output feature map, downsampled if downsample=True.
        """
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinDecoderStage(nn.Module):
    """
    One decoder stage: PatchExpanding upsample, skip-connection concatenation,
    and convolutional refinement.

    Channel arithmetic with concatenation join:
        input (B, dim, ...)
        → PatchExpanding → (B, dim//2, 2x...)
        → concat skip (dim_skip channels) → (B, dim//2 + skip_dim, 2x...)
        → conv refinement → (B, out_dim, 2x...)

    Args:
        dim (int): input channel count from the previous deeper stage
        skip_dim (int): channel count of the matching encoder skip feature map
        out_dim (int): output channel count after conv refinement
        is3d (bool): 3D or 2D mode
        dropout_prob (float): dropout probability in the refinement block
    """

    def __init__(self, dim, skip_dim, out_dim, is3d=True, dropout_prob=0.0):
        super().__init__()
        self.expand = PatchExpanding(dim, is3d=is3d)
        in_channels = dim // 2 + skip_dim
        Conv = nn.Conv3d if is3d else nn.Conv2d
        num_groups_in = _safe_num_groups(in_channels)
        num_groups_out = _safe_num_groups(out_dim)
        self.refine = nn.Sequential(
            nn.GroupNorm(num_groups_in, in_channels),
            Conv(in_channels, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.GroupNorm(num_groups_out, out_dim),
            Conv(out_dim, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, skip):
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature map from the previous (deeper) decoder or bottleneck stage.
        skip : torch.Tensor
            Skip connection from the corresponding encoder stage; must have
            spatial dimensions 2x those of x (before expansion).

        Returns
        -------
        torch.Tensor
            Refined feature map at 2x the spatial resolution of the input.
        """
        x = self.expand(x)
        x = torch.cat([x, skip], dim=1)
        x = self.refine(x)
        return x
