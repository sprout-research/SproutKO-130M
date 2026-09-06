"""Token sampling strategies for autoregressive generation."""

import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.LongTensor:
    """Samples the next token from logits using greedy, temperature, top-k, and top-p strategies.

    Args:
        logits: Unnormalized log probabilities of shape (batch_size, vocab_size).
        temperature: Sampling temperature. Must be >= 0.0. If 0.0, performs deterministic greedy argmax.
        top_k: Number of highest probability tokens to keep. 0 disables top-k filtering.
        top_p: Nucleus cumulative probability threshold in (0.0, 1.0]. 1.0 disables top-p filtering.

    Returns:
        Sampled token tensor of shape (batch_size, 1).
    """
    if logits.ndim != 2:
        raise ValueError(f"Logits must be 2D [B, vocab_size], got shape {logits.shape}")

    if temperature < 0.0:
        raise ValueError(f"Temperature must be non-negative, got {temperature}")
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0.0, 1.0], got {top_p}")

    # Temperature 0.0 is strictly greedy argmax
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Top-K filtering
    if top_k > 0:
        k = min(top_k, scaled_logits.size(-1))
        topk_values, _ = torch.topk(scaled_logits, k=k, dim=-1)
        min_topk = topk_values[..., -1, None]
        scaled_logits = torch.where(scaled_logits < min_topk, float("-inf"), scaled_logits)

    # Top-P (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Shift the indices to keep the first token exceeding the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to the original ordering on a clean boolean mask
        indices_to_remove = torch.zeros_like(scaled_logits, dtype=torch.bool).scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        scaled_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))

    # Convert to probabilities and sample via categorical multinomial
    probs = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
