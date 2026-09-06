"""Causal Language Model (SproutKOForCausalLM) implementation with KV-cache support."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from sproutko.config import ModelConfig
from sproutko.model.cache import KVCache, LayerKVCache
from sproutko.model.rmsnorm import RMSNorm
from sproutko.model.transformer import BackboneOutput, SproutKOModel


@dataclass
class CausalLMOutput:
    """Output dataclass for SproutKOForCausalLM.

    Attributes:
        logits: Next-token prediction logits of shape (batch_size, seq_len, vocab_size).
        past_key_values: Optional KVCache container holding present KV states.
    """

    logits: torch.FloatTensor
    past_key_values: Optional[KVCache] = None


class SproutKOForCausalLM(nn.Module):
    """Causal Language Model with Decoder Backbone and LM Head.

    Architecture:
        Backbone (SproutKOModel) -> LM Head Linear -> Logits [B, T, vocab_size]

    Args:
        config: Model configuration object.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Backbone Transformer
        self.model = SproutKOModel(config)

        # Output projection head (LM Head)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize parameters across all submodules
        self.apply(self._init_weights)
        self._scale_residual_projections()

        # Weight tying: share parameters between token embedding and LM head
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def from_pretrained(
        cls,
        source: Union[str, Path],
        *,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> "SproutKOForCausalLM":
        """Load inference weights from a local export folder or Hugging Face Hub repo.

        A Hub repo id such as ``user/SproutKO-130M`` downloads ``config.json`` and
        ``model.safetensors`` via ``huggingface_hub``. A local directory with the
        same files is loaded offline. This is not ``transformers.AutoModel``.
        """
        from sproutko.pretrained import load_causal_lm, pretrained_hub_kwargs

        return load_causal_lm(source, **pretrained_hub_kwargs(revision, token, cache_dir))

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights according to configuration standard deviation.

        Args:
            module: PyTorch module to initialize.
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _scale_residual_projections(self) -> None:
        """Downscales residual output projections by sqrt(2L) for deep-stack stability."""
        std = self.config.initializer_range / math.sqrt(2.0 * self.config.num_hidden_layers)
        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name.endswith("self_attn.o_proj") or name.endswith("mlp.down_proj"):
                nn.init.normal_(module.weight, mean=0.0, std=std)

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
    ) -> Union[torch.FloatTensor, CausalLMOutput]:
        """Forward pass to compute next-token prediction logits.

        Args:
            input_ids: Token ID tensor of shape (batch_size, seq_len).
            position_ids: Optional position indices of shape (batch_size, seq_len).
            past_key_values: Optional KVCache or tuple of LayerKVCaches.
            use_cache: Whether to return updated KVCache for autoregressive generation.
            inputs_embeds: Optional precomputed embeddings of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask tensor.
            return_dict: Whether to return CausalLMOutput dataclass instead of raw tensor.

        Returns:
            CausalLMOutput if use_cache=True or return_dict=True, else logits tensor.
        """
        backbone_output = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True if (use_cache or return_dict) else False,
        )

        if isinstance(backbone_output, BackboneOutput):
            hidden_states = backbone_output.last_hidden_state
            past_kvs = backbone_output.past_key_values
        else:
            hidden_states = backbone_output
            past_kvs = None

        logits = self.lm_head(hidden_states)

        if use_cache or return_dict:
            return CausalLMOutput(
                logits=logits,
                past_key_values=past_kvs,
            )
        return logits
