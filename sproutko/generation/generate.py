"""Autoregressive generation loop using KV-cache prefill and incremental decode."""

from typing import Optional

import torch
import torch.nn as nn

from sproutko.generation.sampling import sample_next_token


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> torch.LongTensor:
    """Generates tokens autoregressively using KV-cache prefill and single-token decode.

    Args:
        model: SproutKOForCausalLM model instance.
        input_ids: Prompt token IDs tensor of shape (batch_size, prompt_seq_len).
        max_new_tokens: Maximum number of new tokens to generate (>= 0).
        temperature: Sampling temperature (0.0 for deterministic greedy argmax).
        top_k: Top-K token filtering threshold.
        top_p: Top-P nucleus sampling threshold.
        eos_token_id: Optional End-of-Sequence token ID to trigger early stopping.

    Returns:
        Tensor of shape (batch_size, prompt_seq_len + generated_len) containing input and generated tokens.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be a 2D tensor [batch_size, seq_len], got shape {input_ids.shape}")

    batch_size, prompt_len = input_ids.shape
    if prompt_len == 0:
        raise ValueError("Prompt must contain at least 1 token (empty prompt is not supported).")

    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")

    if max_new_tokens == 0:
        return input_ids

    # Validate against context length bound
    max_positions = getattr(getattr(model, "config", None), "max_position_embeddings", 4096)
    if prompt_len + max_new_tokens > max_positions:
        raise ValueError(
            f"Requested total sequence length ({prompt_len + max_new_tokens}) exceeds "
            f"model's max_position_embeddings ({max_positions})."
        )

    was_training = model.training
    model.eval()

    try:
        # -------------------------------------------------------------
        # Phase 1: Prefill prompt sequence (single forward pass)
        # -------------------------------------------------------------
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :]

        # Sample first generated token
        next_token = sample_next_token(
            last_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        generated_tokens = [input_ids, next_token]

        # Early termination tracking
        if eos_token_id is not None:
            active_sequences = (next_token != eos_token_id).squeeze(-1)
            if not active_sequences.any():
                return torch.cat(generated_tokens, dim=-1)
        else:
            active_sequences = None

        # -------------------------------------------------------------
        # Phase 2: Autoregressive decode loop (1 token forward per step)
        # -------------------------------------------------------------
        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if eos_token_id is not None and active_sequences is not None:
                # Finished batch items remain at EOS while unfinished items continue.
                eos_fill = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(active_sequences.unsqueeze(-1), next_token, eos_fill)
                active_sequences = active_sequences & (next_token != eos_token_id).squeeze(-1)
                generated_tokens.append(next_token)
                if not active_sequences.any():
                    break
            else:
                generated_tokens.append(next_token)

        return torch.cat(generated_tokens, dim=-1)

    finally:
        if was_training:
            model.train()
