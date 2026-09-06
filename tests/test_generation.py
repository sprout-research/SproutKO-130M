"""Unit tests for autoregressive generation and sampling strategies."""

import importlib

import pytest
import torch

from sproutko.config import ModelConfig
from sproutko.generation.generate import generate
from sproutko.generation.sampling import sample_next_token
from sproutko.model.causal_lm import SproutKOForCausalLM


@pytest.fixture
def gen_model():
    """Returns an initialized SproutKO model for generation tests."""
    config = ModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    model = SproutKOForCausalLM(config)
    model.eval()
    return model


def test_greedy_generation_determinism_and_shapes(gen_model):
    """Verify greedy generation produces deterministic outputs with correct shapes."""
    prompt = torch.tensor([[5, 12, 42, 7]])
    max_new_tokens = 10

    out1 = generate(gen_model, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    out2 = generate(gen_model, prompt, max_new_tokens=max_new_tokens, temperature=0.0)

    # Determinism
    assert torch.equal(out1, out2)

    # Shape: [B, prompt_len + max_new_tokens]
    assert out1.shape == (1, 4 + max_new_tokens)
    # Prefix matches prompt
    assert torch.equal(out1[:, :4], prompt)


def test_sampling_temperature_semantics():
    """Verify temperature scaling behavior and error handling."""
    logits = torch.randn(2, 50)

    # temperature=0.0 matches greedy argmax
    greedy_token = torch.argmax(logits, dim=-1, keepdim=True)
    temp0_token = sample_next_token(logits, temperature=0.0)
    assert torch.equal(greedy_token, temp0_token)

    # Negative temperature raises ValueError
    with pytest.raises(ValueError, match="Temperature must be non-negative"):
        sample_next_token(logits, temperature=-0.5)


def test_top_k_sampling_behavior():
    """Verify top-k sampling restricts selection to top K candidates."""
    # Create logits where token 7 is highest, token 3 is 2nd highest
    logits = torch.zeros(1, 20)
    logits[0, 7] = 100.0
    logits[0, 3] = 90.0

    # top_k=1 must always pick token 7
    for _ in range(10):
        sampled = sample_next_token(logits, temperature=1.0, top_k=1)
        assert sampled.item() == 7

    # Negative top_k raises ValueError
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        sample_next_token(logits, top_k=-1)


def test_top_p_sampling_validation():
    """Verify top-p threshold validation."""
    logits = torch.randn(1, 20)

    with pytest.raises(ValueError, match="top_p must be in"):
        sample_next_token(logits, top_p=0.0)

    with pytest.raises(ValueError, match="top_p must be in"):
        sample_next_token(logits, top_p=1.5)


def test_eos_token_early_stopping(gen_model):
    """Verify generation halts early when EOS token is generated."""
    prompt = torch.tensor([[1, 2, 3]])

    # Find the token that model naturally predicts after prompt
    with torch.no_grad():
        natural_first_token = gen_model(prompt).argmax(dim=-1)[:, -1].item()

    # Set that token as eos_token_id
    out = generate(gen_model, prompt, max_new_tokens=20, temperature=0.0, eos_token_id=natural_first_token)

    # Should have stopped after generating the first EOS token (length 3 + 1 = 4)
    assert out.shape == (1, 4)
    assert out[0, -1].item() == natural_first_token


def test_empty_prompt_and_overflow_validation(gen_model):
    """Verify empty prompt and context length overflow are rejected with clear errors."""
    empty_prompt = torch.empty((1, 0), dtype=torch.long)
    with pytest.raises(ValueError, match="Prompt must contain at least 1 token"):
        generate(gen_model, empty_prompt, max_new_tokens=5)

    # Max position embeddings for gen_model is 64
    long_prompt = torch.randint(0, 100, (1, 50))
    with pytest.raises(ValueError, match="exceeds model's max_position_embeddings"):
        generate(gen_model, long_prompt, max_new_tokens=20)  # 50 + 20 = 70 > 64


def test_generation_uses_incremental_decoding(gen_model):
    """Verify that generation forwards 1 token per decode step instead of re-forwarding full prompt."""
    forward_input_lengths = []

    original_forward = gen_model.forward

    def instrumented_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is not None:
            forward_input_lengths.append(input_ids.shape[1])
        return original_forward(*args, **kwargs)

    gen_model.forward = instrumented_forward
    prompt = torch.tensor([[10, 20, 30]])
    max_new_tokens = 4

    try:
        generate(gen_model, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        # Prefill should receive seq_len=3, followed by 3 decode steps of seq_len=1 each
        assert forward_input_lengths == [3, 1, 1, 1], f"Unexpected forward input lengths: {forward_input_lengths}"
    finally:
        gen_model.forward = original_forward


def test_finished_batch_items_stay_at_eos(monkeypatch, gen_model):
    calls = 0

    def staged_sample(logits, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor([[2], [7]], device=logits.device)
        if calls == 2:
            return torch.tensor([[9], [2]], device=logits.device)
        return torch.tensor([[9], [9]], device=logits.device)

    generate_module = importlib.import_module("sproutko.generation.generate")
    monkeypatch.setattr(generate_module, "sample_next_token", staged_sample)
    prompt = torch.tensor([[1, 3], [1, 4]])
    output = generate(gen_model, prompt, max_new_tokens=4, eos_token_id=2)
    assert output[:, 2:].tolist() == [[2, 2], [7, 2]]
