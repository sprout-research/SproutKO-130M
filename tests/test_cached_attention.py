"""Critical Invariant Tests: Numerical equivalence between KV-cache and full reference forward."""

import pytest
import torch

from sproutko.config import ModelConfig
from sproutko.model.causal_lm import SproutKOForCausalLM


@pytest.fixture
def test_model():
    """Returns an initialized SproutKO model for testing."""
    config = ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=256,
    )
    model = SproutKOForCausalLM(config)
    model.eval()
    return model


def test_prefill_numerical_equivalence(test_model):
    """Verify that cached prefill produces numerically identical logits to non-cached full forward."""
    input_ids = torch.randint(0, 256, (2, 8))

    with torch.no_grad():
        full_logits = test_model(input_ids, use_cache=False)
        prefill_out = test_model(input_ids, use_cache=True)

    cached_logits = prefill_out.logits
    max_diff = (full_logits - cached_logits).abs().max().item()

    assert max_diff < 1e-6, f"Prefill logits differ from full forward! Max diff: {max_diff}"
    assert torch.allclose(full_logits, cached_logits, atol=1e-6, rtol=1e-5)


def test_single_step_decode_numerical_equivalence(test_model):
    """Critical Test: Full forward logits(D) == Cached decode logits(D)."""
    # Sequence [A, B, C, D]
    full_sequence = torch.randint(0, 256, (2, 4))
    prompt = full_sequence[:, :3]  # [A, B, C]
    next_token = full_sequence[:, 3:4]  # [D]

    with torch.no_grad():
        full_out = test_model(full_sequence, use_cache=False)
        prefill_out = test_model(prompt, use_cache=True)
        decode_out = test_model(next_token, past_key_values=prefill_out.past_key_values, use_cache=True)

    expected_d_logits = full_out[:, 3, :]
    actual_d_logits = decode_out.logits[:, 0, :]

    max_diff = (expected_d_logits - actual_d_logits).abs().max().item()
    assert max_diff < 1e-5, f"1-step decode logits mismatch! Max diff: {max_diff}"
    assert torch.allclose(expected_d_logits, actual_d_logits, atol=1e-5, rtol=1e-4)


def test_multi_step_decode_numerical_equivalence(test_model):
    """Critical Test: Multi-step decode produces exact logits at each decode step."""
    # Sequence [A, B, C, D, E, F, G]
    seq_len = 7
    full_sequence = torch.randint(0, 256, (2, seq_len))
    prompt = full_sequence[:, :3]  # [A, B, C]

    with torch.no_grad():
        full_out = test_model(full_sequence, use_cache=False)

        # Prefill on [A, B, C]
        curr_out = test_model(prompt, use_cache=True)
        cache = curr_out.past_key_values

        # Incrementally decode tokens D (idx 3), E (idx 4), F (idx 5), G (idx 6)
        for t in range(3, seq_len):
            step_token = full_sequence[:, t : t + 1]
            decode_out = test_model(step_token, past_key_values=cache, use_cache=True)
            cache = decode_out.past_key_values

            expected_step_logits = full_out[:, t, :]
            actual_step_logits = decode_out.logits[:, 0, :]

            max_diff = (expected_step_logits - actual_step_logits).abs().max().item()
            assert max_diff < 1e-5, f"Mismatch at step {t}! Max diff: {max_diff}"
            assert torch.allclose(expected_step_logits, actual_step_logits, atol=1e-5, rtol=1e-4)


def test_multi_token_append_numerical_equivalence(test_model):
    """Verify multi-token cached append [D, E] produces exact logits for both tokens."""
    # Sequence [A, B, C, D, E]
    full_sequence = torch.randint(0, 256, (2, 5))
    prompt = full_sequence[:, :3]  # [A, B, C]
    append_tokens = full_sequence[:, 3:]  # [D, E]

    with torch.no_grad():
        full_out = test_model(full_sequence, use_cache=False)
        prefill_out = test_model(prompt, use_cache=True)
        append_out = test_model(append_tokens, past_key_values=prefill_out.past_key_values, use_cache=True)

    expected_append_logits = full_out[:, 3:, :]
    actual_append_logits = append_out.logits

    max_diff = (expected_append_logits - actual_append_logits).abs().max().item()
    assert max_diff < 1e-5, f"Multi-token append logits mismatch! Max diff: {max_diff}"
    assert torch.allclose(expected_append_logits, actual_append_logits, atol=1e-5, rtol=1e-4)


def test_batch_size_greater_than_one_equivalence(test_model):
    """Verify KV-cache equivalence across batch_size=4."""
    batch_size = 4
    full_sequence = torch.randint(0, 256, (batch_size, 6))

    with torch.no_grad():
        full_out = test_model(full_sequence, use_cache=False)
        prefill_out = test_model(full_sequence[:, :4], use_cache=True)
        decode_out = test_model(full_sequence[:, 4:5], past_key_values=prefill_out.past_key_values, use_cache=True)

    expected = full_out[:, 4, :]
    actual = decode_out.logits[:, 0, :]
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)


def test_explicit_position_ids_with_cache(test_model):
    """Verify explicit position_ids with KV cache behaves correctly."""
    prompt = torch.tensor([[10, 20, 30]])
    next_tok = torch.tensor([[40]])

    with torch.no_grad():
        prefill = test_model(prompt, position_ids=torch.tensor([[0, 1, 2]]), use_cache=True)
        decode = test_model(
            next_tok, position_ids=torch.tensor([[3]]), past_key_values=prefill.past_key_values, use_cache=True
        )
        full = test_model(torch.tensor([[10, 20, 30, 40]]), position_ids=torch.tensor([[0, 1, 2, 3]]), use_cache=False)

    assert torch.allclose(full[:, 3, :], decode.logits[:, 0, :], atol=1e-5, rtol=1e-4)
