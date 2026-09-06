"""Unit tests for ModelConfig validation and presets."""

import pytest

from sproutko.config import ModelConfig


def test_valid_default_config():
    """Test that default configuration passes validation."""
    config = ModelConfig()
    assert config.vocab_size == 32_000
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 16
    assert config.num_attention_heads == 12
    assert config.num_key_value_heads == 4
    assert config.head_dim == 64
    assert config.num_key_value_groups == 3


def test_preset_sproutko_130m():
    """Test SproutKO-130M preset."""
    config = ModelConfig.sproutko_130m()
    assert config.hidden_size == 768
    assert config.num_attention_heads * config.head_dim == 768
    assert config.num_attention_heads % config.num_key_value_heads == 0
    assert config.tie_word_embeddings is True
    assert config.num_hidden_layers == 16
    assert config.intermediate_size == 2176


def test_scaling_presets():
    """Test all scaling presets instantiate without validation errors."""
    presets = [
        ModelConfig.sproutko_130m(),
        ModelConfig.sproutko_300m(),
        ModelConfig.sproutko_1b(),
        ModelConfig.sproutko_7b(),
        ModelConfig.sproutko_30b(),
    ]
    for cfg in presets:
        assert cfg.hidden_size == cfg.num_attention_heads * cfg.head_dim
        assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
        assert cfg.head_dim % 2 == 0


def test_invalid_hidden_size_mismatch():
    """Test that hidden_size mismatching num_attention_heads * head_dim raises ValueError."""
    with pytest.raises(ValueError, match="hidden_size .* must equal num_attention_heads .* head_dim"):
        ModelConfig(
            hidden_size=1024,
            num_attention_heads=12,
            head_dim=64,  # 12 * 64 = 768 != 1024
        )


def test_invalid_gqa_grouping():
    """Test that non-divisible query / kv heads raise ValueError."""
    with pytest.raises(ValueError, match="num_attention_heads .* must be divisible by num_key_value_heads"):
        ModelConfig(
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=5,  # 12 % 5 != 0
            head_dim=64,
        )


def test_odd_head_dim_rejection():
    """Test that odd head_dim is rejected for RoPE compatibility."""
    with pytest.raises(ValueError, match="head_dim must be even"):
        ModelConfig(
            hidden_size=63,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=63,
        )


def test_non_positive_values():
    """Test that zero or negative dimensions raise ValueError."""
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=0)
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=-768)
    with pytest.raises(ValueError):
        ModelConfig(num_hidden_layers=0)
    with pytest.raises(ValueError):
        ModelConfig(rms_norm_eps=-1e-5)
    with pytest.raises(ValueError):
        ModelConfig(rope_theta=0.0)
    with pytest.raises(ValueError):
        ModelConfig(initializer_range=0.0)


def test_serialization():
    """Test dictionary conversion roundtrip."""
    config = ModelConfig.sproutko_130m()
    config_dict = config.to_dict()
    restored_config = ModelConfig.from_dict(config_dict)
    assert config == restored_config
