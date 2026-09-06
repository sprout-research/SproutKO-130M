"""Rotary Positional Embedding (RoPE) implementation."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor.

    Args:
        x: Input tensor of shape (..., head_dim).

    Returns:
        Tensor with the first half negated and swapped with the second half:
        [-x2, x1] where x = [x1, x2].
    """
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Positional Embedding to query and key tensors.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
        k: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim).
        cos: Cosine component of shape (batch_size, 1, seq_len, head_dim).
        sin: Sine component of shape (batch_size, 1, seq_len, head_dim).

    Returns:
        Tuple of (rotated_q, rotated_k) with identical shapes to inputs.
    """
    q_fp32 = q.float()
    k_fp32 = k.float()
    cos_fp32 = cos.float()
    sin_fp32 = sin.float()
    q_embed = (q_fp32 * cos_fp32) + (rotate_half(q_fp32) * sin_fp32)
    k_embed = (k_fp32 * cos_fp32) + (rotate_half(k_fp32) * sin_fp32)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) generator.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)

    Args:
        dim: Dimensionality of each attention head (head_dim). Must be even.
        max_position_embeddings: Maximum sequence length supported.
        base: Base frequency for geometric progression (rope_theta).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Rotary embedding dim must be even, got {dim}")

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute inverse frequencies: theta_i = base^(-2i/dim) for i in [0, dim/2)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cosine and sine components for given position IDs or sequence length.

        Args:
            x: Example tensor to extract device and dtype from, shape (batch_size, ..., seq_len, head_dim).
            position_ids: Optional position indices of shape (batch_size, seq_len).
            seq_len: Optional sequence length to use if position_ids is None.

        Returns:
            Tuple of (cos, sin) tensors of shape (batch_size, 1, seq_len, head_dim).
        """
        device = x.device

        if position_ids is None:
            if seq_len is None:
                seq_len = x.shape[-2]
            batch_size = x.shape[0]
            # Generate default contiguous position IDs [0, 1, ..., seq_len - 1]
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        else:
            batch_size, seq_len = position_ids.shape

        # position_ids: [B, T] -> [B, T, 1]
        pos = position_ids.float().unsqueeze(-1)
        # inv_freq: [D // 2] -> [1, 1, D // 2]
        inv_freq = self.inv_freq.to(device).unsqueeze(0).unsqueeze(0)

        # freqs: [B, T, D // 2]
        freqs = pos * inv_freq
        # emb: [B, T, D] = concat([freqs, freqs], dim=-1)
        emb = torch.cat((freqs, freqs), dim=-1)

        # cos, sin: [B, 1, T, D] in float32; apply_rotary_pos_emb downcasts after the rotation
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)

        return cos, sin

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_position_embeddings={self.max_position_embeddings}, base={self.base}"
