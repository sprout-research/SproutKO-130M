"""Decoder-only Transformer backbone implementation with KV-cache support."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from sproutko.config import ModelConfig
from sproutko.model.block import TransformerBlock
from sproutko.model.cache import KVCache, LayerKVCache
from sproutko.model.rmsnorm import RMSNorm


@dataclass
class BackboneOutput:
    """Output dataclass for SproutKOModel backbone.

    Attributes:
        last_hidden_state: Output hidden states tensor of shape (batch_size, seq_len, hidden_size).
        past_key_values: Optional KVCache container holding present KV states.
    """

    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[KVCache] = None


class SproutKOModel(nn.Module):
    """Decoder-only Transformer backbone for SproutKO models.

    Components:
        embed_tokens -> TransformerBlock x N -> final_norm

    Args:
        config: Model configuration object.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Token embedding lookup table
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.gradient_checkpointing = False

        # Transformer blocks stack
        self.layers = nn.ModuleList([TransformerBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        # Final RMS LayerNorm
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[
            Union[KVCache, Tuple[Union[LayerKVCache, Tuple[torch.Tensor, torch.Tensor]], ...]]
        ] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.FloatTensor, BackboneOutput]:
        """Forward pass through the Transformer backbone with optional KV-cache.

        Args:
            input_ids: Token ID tensor of shape (batch_size, seq_len).
            position_ids: Optional position indices of shape (batch_size, seq_len).
            past_key_values: Optional KVCache or tuple of LayerKVCaches.
            use_cache: Whether to return updated KVCache.
            inputs_embeds: Optional precomputed embeddings of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask tensor.
            return_dict: Whether to return BackboneOutput dataclass instead of raw tensor.

        Returns:
            BackboneOutput if use_cache=True or return_dict=True, else hidden_states tensor.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds at the same time.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds.")

        if inputs_embeds is None:
            batch_size, seq_len = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
            device = input_ids.device
        else:
            batch_size, seq_len, _ = inputs_embeds.shape
            device = inputs_embeds.device

        # Convert past_key_values to unified KVCache format if provided
        cache_obj: Optional[KVCache] = None
        past_seq_len = 0
        if past_key_values is not None:
            cache_obj = KVCache.from_legacy_tuple(past_key_values)
            past_seq_len = cache_obj.get_seq_len(0)

        # Generate position_ids if not explicitly supplied
        if position_ids is None:
            position_ids = (
                torch.arange(
                    past_seq_len,
                    past_seq_len + seq_len,
                    dtype=torch.long,
                    device=device,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        hidden_states = inputs_embeds
        present_key_values = KVCache(num_layers=self.config.num_hidden_layers) if use_cache else None
        shared_cos, shared_sin = self.layers[0].self_attn.rotary_emb(
            hidden_states,
            position_ids=position_ids,
            seq_len=None,
        )

        # Pass through all Transformer blocks
        use_checkpoint = bool(self.gradient_checkpointing and self.training and not use_cache)
        for i, layer in enumerate(self.layers):
            layer_past = cache_obj[i] if cache_obj is not None and i < len(cache_obj) else None

            if use_cache:
                hidden_states, present_layer = layer(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    use_cache=True,
                    attention_mask=attention_mask,
                    cos=shared_cos,
                    sin=shared_sin,
                )
                assert present_key_values is not None
                present_key_values.layers[i] = present_layer
            elif use_checkpoint:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    position_ids,
                    layer_past,
                    False,
                    attention_mask,
                    shared_cos,
                    shared_sin,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    use_cache=False,
                    attention_mask=attention_mask,
                    cos=shared_cos,
                    sin=shared_sin,
                )

        # Apply final normalization
        hidden_states = self.final_norm(hidden_states)

        if use_cache or return_dict:
            return BackboneOutput(
                last_hidden_state=hidden_states,
                past_key_values=present_key_values,
            )
        return hidden_states
