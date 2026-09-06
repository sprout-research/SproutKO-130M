"""Grouped Query Attention (GQA) with KV-cache and causal masking implementation."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sproutko.config import ModelConfig
from sproutko.model.cache import LayerKVCache
from sproutko.model.rope import RotaryEmbedding, apply_rotary_pos_emb


def _prepare_attention_bias(
    attention_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    q_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Normalizes keep masks or additive masks to ``[B, 1, Q, K]`` semantics.

    Boolean and binary 0/1 masks use ``1/True = attend`` and ``0/False = mask``
    in 2D, 3D, and 4D. Other floating tensors are additive attention bias.
    """
    if attention_mask is None:
        return None
    mask = attention_mask.to(device=device)
    if mask.ndim == 2:
        if tuple(mask.shape) != (batch_size, kv_len):
            raise ValueError(f"2D attention_mask must have shape {(batch_size, kv_len)}, got {tuple(mask.shape)}")
        if mask.dtype != torch.bool and not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("2D attention_mask values must be boolean or binary 0/1 keep indicators")
        keep = mask.to(torch.bool)
        bias = torch.zeros((batch_size, 1, 1, kv_len), dtype=dtype, device=device)
        return bias.masked_fill(~keep[:, None, None, :], float("-inf"))
    if mask.ndim == 3:
        if mask.shape[0] != batch_size or mask.shape[-2:] != (q_len, kv_len):
            raise ValueError(
                f"3D attention_mask must have shape {(batch_size, q_len, kv_len)}, got {tuple(mask.shape)}"
            )
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4:
        if (
            mask.shape[0] not in (1, batch_size)
            or mask.shape[1] != 1
            or mask.shape[-2] not in (1, q_len)
            or mask.shape[-1] != kv_len
        ):
            raise ValueError(
                f"4D attention_mask must be broadcastable to {(batch_size, 1, q_len, kv_len)}, got {tuple(mask.shape)}"
            )
    else:
        raise ValueError(f"attention_mask must be 2D, 3D, or 4D, got {mask.ndim}D")

    if mask.dtype == torch.bool or torch.all((mask == 0) | (mask == 1)):
        keep = mask.to(torch.bool)
        bias = torch.zeros(mask.shape, dtype=dtype, device=device)
        return bias.masked_fill(~keep, float("-inf"))
    return mask.to(dtype=dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key or value heads to match the number of query heads in GQA.

    Transforms shape:
        (batch_size, num_kv_heads, seq_len, head_dim)
        -> (batch_size, num_kv_heads * n_rep, seq_len, head_dim)

    Args:
        hidden_states: Tensor of shape (batch_size, num_kv_heads, seq_len, head_dim).
        n_rep: Repeating factor (num_attention_heads // num_key_value_heads).

    Returns:
        Tensor with repeated heads matching query heads dimension.
    """
    if n_rep == 1:
        return hidden_states

    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    # Expand along a new group dimension without copying data
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
    # Reshape to merge num_kv_heads and n_rep into total attention heads
    return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)


class CausalSelfAttention(nn.Module):
    """Grouped Query Attention (GQA) module with KV-cache and causal masking.

    Reference: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023)

    Args:
        config: Model configuration object.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_key_value_groups
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Query, Key, Value, Output linear projections
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # Rotary Positional Embedding generator
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Union[LayerKVCache, Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[LayerKVCache]]]:
        """Forward pass for Grouped Query Causal Self-Attention with optional KV-cache.

        Args:
            hidden_states: Input tensor of shape (batch_size, q_len, hidden_size).
            position_ids: Optional token position indices of shape (batch_size, q_len).
            past_key_value: Optional cached Key and Value states from prior steps.
            use_cache: Whether to return updated LayerKVCache for subsequent decode steps.
            attention_mask: Optional attention mask tensor.
            cos: Optional precomputed RoPE cosine, shared across layers.
            sin: Optional precomputed RoPE sine, shared across layers.

        Returns:
            Tuple of:
                - output tensor of shape (batch_size, q_len, hidden_size)
                - present_key_value: LayerKVCache holding (key, value) if use_cache=True, else None
        """
        batch_size, q_len, _ = hidden_states.shape

        # Step 1: Linear projections
        # query: [B, q_len, num_attention_heads * head_dim]
        # key:   [B, q_len, num_key_value_heads * head_dim]
        # value: [B, q_len, num_key_value_heads * head_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Step 2: Reshape to multi-head format [B, H, q_len, D]
        query_states = query_states.view(batch_size, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Determine past sequence length from cache
        if past_key_value is not None:
            if isinstance(past_key_value, LayerKVCache):
                past_seq_len = past_key_value.seq_len
                past_k, past_v = past_key_value.key, past_key_value.value
            else:
                past_k, past_v = past_key_value[0], past_key_value[1]
                past_seq_len = past_k.shape[2]
        else:
            past_seq_len = 0
            past_k, past_v = None, None

        # Step 3: Compute and apply Rotary Positional Embeddings to current Q and K
        if position_ids is None:
            position_ids = (
                torch.arange(
                    past_seq_len,
                    past_seq_len + q_len,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        if cos is None or sin is None:
            cos, sin = self.rotary_emb(value_states, position_ids=position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Step 4: KV-Cache append (RoPE is applied ONCE before caching)
        max_cache_len = int(self.config.max_position_embeddings)
        if isinstance(past_key_value, LayerKVCache) and use_cache:
            past_key_value.append(key_states, value_states)
            key_states = past_key_value.key
            value_states = past_key_value.value
            present_key_value = past_key_value
        else:
            if past_k is not None and past_v is not None:
                key_states = torch.cat([past_k, key_states], dim=2)
                value_states = torch.cat([past_v, value_states], dim=2)
            present_key_value = (
                LayerKVCache.with_capacity(key_states, value_states, max_cache_len) if use_cache else None
            )

        kv_len = key_states.shape[2]

        attn_bias = _prepare_attention_bias(
            attention_mask,
            batch_size=batch_size,
            q_len=q_len,
            kv_len=kv_len,
            dtype=query_states.dtype,
            device=hidden_states.device,
        )
        is_causal_prefill = past_seq_len == 0 and q_len == kv_len and q_len > 1
        if q_len > 1 and (not is_causal_prefill or attention_mask is not None):
            query_positions = torch.arange(q_len, device=hidden_states.device).unsqueeze(1) + past_seq_len
            key_positions = torch.arange(kv_len, device=hidden_states.device).unsqueeze(0)
            causal_mask = torch.zeros((q_len, kv_len), device=hidden_states.device, dtype=query_states.dtype)
            causal_mask = causal_mask.masked_fill(key_positions > query_positions, float("-inf"))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_bias = causal_mask if attn_bias is None else attn_bias + causal_mask
            is_causal_prefill = False

        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
        value_states_rep = repeat_kv(value_states, self.num_key_value_groups)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = None
            if self.num_key_value_groups > 1:
                try:
                    attn_output = F.scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        attn_mask=attn_bias,
                        dropout_p=0.0,
                        is_causal=is_causal_prefill,
                        enable_gqa=True,
                    )
                except TypeError:
                    attn_output = None
            if attn_output is None:
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states_rep,
                    value_states_rep,
                    attn_mask=attn_bias,
                    dropout_p=0.0,
                    is_causal=is_causal_prefill,
                )
        else:
            attn_scores = torch.matmul(query_states, key_states_rep.transpose(-1, -2)) * self.scale
            if is_causal_prefill:
                query_positions = torch.arange(q_len, device=hidden_states.device).unsqueeze(1)
                key_positions = torch.arange(kv_len, device=hidden_states.device).unsqueeze(0)
                causal_mask = torch.zeros((q_len, kv_len), device=hidden_states.device, dtype=attn_scores.dtype)
                causal_mask = causal_mask.masked_fill(key_positions > query_positions, float("-inf"))
                attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
            elif attn_bias is not None:
                attn_scores = attn_scores + attn_bias
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states_rep)

        # Transpose and reshape back to [B, q_len, num_attention_heads * head_dim]
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.num_attention_heads * self.head_dim)
        )

        # Step 11: Final output projection to [B, q_len, hidden_size]
        output = self.o_proj(attn_output)
        if use_cache:
            return output, present_key_value
        return output
