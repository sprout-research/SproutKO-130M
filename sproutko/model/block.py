"""Transformer Block with Pre-Norm, residual connections, and KV-cache support."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from sproutko.config import ModelConfig
from sproutko.model.attention import CausalSelfAttention
from sproutko.model.cache import LayerKVCache
from sproutko.model.mlp import SwiGLU
from sproutko.model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """Single Decoder-only Transformer layer with Pre-Norm architecture and KV-cache support.

    Structure:
        x -> RMSNorm -> CausalSelfAttention -> + Residual (x)
          -> RMSNorm -> SwiGLU              -> + Residual

    Args:
        config: Model configuration object.
        layer_idx: Optional index of the layer in the model stack.
    """

    def __init__(self, config: ModelConfig, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Attention block with Pre-Norm
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)

        # Feed-forward block with Pre-Norm
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)

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
        """Forward pass for a single Transformer block.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            position_ids: Optional position indices of shape (batch_size, seq_len).
            past_key_value: Optional cached Key/Value states from previous generation steps.
            use_cache: Whether to return updated LayerKVCache for this block.
            attention_mask: Optional attention mask.

        Returns:
            If use_cache is True:
                Tuple of (output_hidden_states, present_key_value)
            If use_cache is False:
                output_hidden_states
        """
        # Pre-Norm Attention with Residual Connection
        residual = hidden_states
        normed_attn_input = self.attention_norm(hidden_states)
        if use_cache:
            attn_output, present_key_value = self.self_attn(
                hidden_states=normed_attn_input,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=True,
                attention_mask=attention_mask,
                cos=cos,
                sin=sin,
            )
        else:
            attn_output = self.self_attn(
                hidden_states=normed_attn_input,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=False,
                attention_mask=attention_mask,
                cos=cos,
                sin=sin,
            )
            present_key_value = None
        hidden_states = residual + attn_output

        # Pre-Norm SwiGLU MLP with Residual Connection
        residual = hidden_states
        normed_mlp_input = self.ffn_norm(hidden_states)
        mlp_output = self.mlp(normed_mlp_input)
        hidden_states = residual + mlp_output

        if use_cache:
            return hidden_states, present_key_value
        return hidden_states
