"""Building blocks for the Transolver Physics-Attention Transformer."""

import torch
import torch.nn as nn


class PhysicsAttention(nn.Module):
    """
    Physics-Attention from Transolver (Wu et al., 2024).

    Instead of attending over all N mesh points (O(N^2)), this module:
      1. Learns per-point soft assignments M ∈ R^{N × n_slices} via a
         linear projection followed by a softmax over the slice dimension.
      2. Aggregates: T = M^T @ x  →  (B, n_slices, C)
      3. Runs standard multi-head self-attention among the n_slices tokens.
      4. Scatters back: out = M @ T'  →  (B, N, C)

    Attention complexity is O(n_slices^2) instead of O(N^2).

    Args:
        embed_dim (int): channel dimension C
        n_slices (int): number of physics-state slices (default: 64)
        num_heads (int): number of attention heads for the slice-level attention
        qkv_bias (bool): add learnable bias in the QKV projection
        attn_dropout (float): dropout on attention weights
        proj_dropout (float): dropout after the output projection
    """

    def __init__(
        self,
        embed_dim: int,
        n_slices: int = 64,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.n_slices = n_slices

        # Per-point slice weight projection: C -> n_slices, then softmax over slices
        self.slice_proj = nn.Linear(embed_dim, n_slices, bias=True)

        # Slice-level multi-head self-attention (batch_first=True uses (B, N, C) convention)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            bias=qkv_bias,
            batch_first=True,
        )

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, N, C).

        Returns
        -------
        torch.Tensor
            Shape (B, N, C).
        """
        # Step 1: Per-point slice weights  →  (B, N, n_slices)
        M = torch.softmax(self.slice_proj(x), dim=-1)

        # Step 2: Aggregate points into physics-state tokens  →  (B, n_slices, C)
        # T[b, s, c] = sum_n  M[b, n, s] * x[b, n, c]
        T = torch.einsum('bns,bnc->bsc', M, x)

        # Step 3: Self-attention on n_slices tokens
        T_prime, _ = self.attn(T, T, T)  # (B, n_slices, C)

        # Step 4: Scatter back to per-point representation  →  (B, N, C)
        # out[b, n, c] = sum_s  M[b, n, s] * T_prime[b, s, c]
        out = torch.einsum('bns,bsc->bnc', M, T_prime)

        out = self.proj_drop(self.proj(out))
        return out


class TransolverBlock(nn.Module):
    """
    One Transolver block: pre-norm PhysicsAttention + pre-norm MLP.

    Block structure (pre-norm):
        x = x + PhysicsAttention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Operates on sequence tensors of shape (B, N, C).

    Args:
        embed_dim (int): channel dimension C
        n_slices (int): number of physics-state slices
        num_heads (int): attention heads for slice attention
        mlp_ratio (float): MLP hidden dimension = embed_dim * mlp_ratio
        qkv_bias (bool): learnable bias in attention QKV
        dropout_prob (float): dropout in MLP and after output projection
        attn_dropout (float): dropout on attention weights
    """

    def __init__(
        self,
        embed_dim: int,
        n_slices: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout_prob: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = PhysicsAttention(
            embed_dim=embed_dim,
            n_slices=n_slices,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout_prob,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, N, C).

        Returns
        -------
        torch.Tensor
            Shape (B, N, C).
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
