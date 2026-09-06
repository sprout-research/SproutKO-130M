"""Unit tests for KV-Cache data structures, shapes, and validation."""

import pytest
import torch

from sproutko.config import ModelConfig
from sproutko.model.cache import KVCache, LayerKVCache
from sproutko.model.causal_lm import SproutKOForCausalLM


def test_layer_kv_cache_shapes_and_unrepeated_heads():
    """Verify LayerKVCache stores only unrepeated KV heads [B, num_kv_heads, T, head_dim]."""
    batch_size = 2
    num_kv_heads = 4
    seq_len = 8
    head_dim = 64

    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

    layer_cache = LayerKVCache(key=key, value=value)
    assert layer_cache.key.shape == (batch_size, num_kv_heads, seq_len, head_dim)
    assert layer_cache.value.shape == (batch_size, num_kv_heads, seq_len, head_dim)
    assert layer_cache.seq_len == seq_len
    assert layer_cache.num_kv_heads == num_kv_heads
    assert layer_cache.head_dim == head_dim


def test_kv_cache_growth_across_prefill_and_decode():
    """Verify that KV cache grows incrementally across prefill and decode steps."""
    config = ModelConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    model = SproutKOForCausalLM(config)
    model.eval()

    # Step 1: Prefill with prompt length 4
    prompt = torch.tensor([[10, 20, 30, 40]])
    with torch.no_grad():
        out_prefill = model(prompt, use_cache=True)

    cache = out_prefill.past_key_values
    assert isinstance(cache, KVCache)
    assert len(cache) == config.num_hidden_layers
    for layer_idx in range(config.num_hidden_layers):
        layer = cache[layer_idx]
        assert layer is not None
        assert layer.key.shape == (1, config.num_key_value_heads, 4, config.head_dim)
        assert layer.value.shape == (1, config.num_key_value_heads, 4, config.head_dim)

    # Step 2: Single-token decode step #1
    token1 = torch.tensor([[50]])
    with torch.no_grad():
        out_decode1 = model(token1, past_key_values=cache, use_cache=True)

    cache1 = out_decode1.past_key_values
    assert isinstance(cache1, KVCache)
    for layer_idx in range(config.num_hidden_layers):
        layer = cache1[layer_idx]
        assert layer is not None
        assert layer.seq_len == 5
        assert layer.key.shape == (1, config.num_key_value_heads, 5, config.head_dim)

    # Step 3: Single-token decode step #2
    token2 = torch.tensor([[60]])
    with torch.no_grad():
        out_decode2 = model(token2, past_key_values=cache1, use_cache=True)

    cache2 = out_decode2.past_key_values
    assert isinstance(cache2, KVCache)
    for layer_idx in range(config.num_hidden_layers):
        layer = cache2[layer_idx]
        assert layer is not None
        assert layer.seq_len == 6
        assert layer.key.shape == (1, config.num_key_value_heads, 6, config.head_dim)


def test_layer_kv_cache_validation_errors():
    """Verify invalid LayerKVCache instantiation raises ValueError."""
    # Key / Value shape mismatch
    k = torch.randn(2, 4, 8, 64)
    v = torch.randn(2, 4, 7, 64)  # Mismatched length
    with pytest.raises(ValueError, match="Key and Value cache shapes must match"):
        LayerKVCache(key=k, value=v)

    # Non-4D tensor
    k_3d = torch.randn(2, 8, 64)
    v_3d = torch.randn(2, 8, 64)
    with pytest.raises(ValueError, match="must be 4D"):
        LayerKVCache(key=k_3d, value=v_3d)


def test_kv_cache_container_validation():
    """Verify KVCache validation detects inconsistent layers and shapes."""
    cache = KVCache(num_layers=2)
    k1 = torch.randn(1, 4, 10, 64)
    v1 = torch.randn(1, 4, 10, 64)
    cache.layers[0] = LayerKVCache(key=k1, value=v1)

    # Inconsistent sequence length in layer 1
    k2 = torch.randn(1, 4, 8, 64)
    v2 = torch.randn(1, 4, 8, 64)
    cache.layers[1] = LayerKVCache(key=k2, value=v2)

    with pytest.raises(ValueError, match="Inconsistent sequence lengths"):
        cache.validate(expected_num_layers=2, expected_num_kv_heads=4, expected_head_dim=64)


def test_kv_cache_legacy_tuple_conversion():
    """Verify seamless conversion from legacy tuple to KVCache."""
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)
    legacy_tuple = ((k, v), (k, v))

    cache = KVCache.from_legacy_tuple(legacy_tuple)
    assert len(cache) == 2
    assert cache[0].seq_len == 4
    assert cache[1].seq_len == 4


def test_kv_cache_dtype_and_device_preservation():
    """Verify that KV cache preserves dtype without forced float32 casting."""
    k = torch.randn(1, 2, 4, 16, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 4, 16, dtype=torch.bfloat16)
    layer = LayerKVCache(key=k, value=v)
    assert layer.key.dtype == torch.bfloat16
    assert layer.value.dtype == torch.bfloat16


def test_layer_kv_cache_append_uses_preallocated_buffer():
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)
    cache = LayerKVCache.with_capacity(key, value, max_seq_len=16)
    assert cache.seq_len == 4
    assert cache.key.shape == (1, 2, 4, 8)
    extra_k = torch.randn(1, 2, 1, 8)
    extra_v = torch.randn(1, 2, 1, 8)
    cache.append(extra_k, extra_v)
    assert cache.seq_len == 5
    assert cache.key.shape == (1, 2, 5, 8)
    assert torch.equal(cache.key[:, :, :4], key)
    assert torch.equal(cache.key[:, :, 4:], extra_k)
