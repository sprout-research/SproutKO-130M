"""Model configuration for the SproutKO decoder-only Transformer family."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Configuration class for SproutKO Transformer models.

    Attributes:
        vocab_size: Size of the vocabulary (default: 32,000).
        hidden_size: Dimensionality of the model embeddings and hidden states.
        num_hidden_layers: Number of Transformer blocks.
        num_attention_heads: Number of attention query heads.
        num_key_value_heads: Number of attention key/value heads for GQA.
        head_dim: Dimensionality of each attention head.
        intermediate_size: Dimensionality of the SwiGLU MLP intermediate layer.
        max_position_embeddings: Maximum sequence length supported by RoPE embeddings.
        rope_theta: Base period for Rotary Positional Embeddings.
        rms_norm_eps: Epsilon value added to denominator in RMSNorm for numerical stability.
        attention_bias: Whether to include bias parameters in attention projections.
        mlp_bias: Whether to include bias parameters in MLP projections.
        tie_word_embeddings: Whether to share weights between token embedding and LM head.
        initializer_range: Standard deviation for weight initialization.
    """

    vocab_size: int = 32_000
    hidden_size: int = 768
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 2176
    max_position_embeddings: int = 4096
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        """Validate configuration parameters for architectural correctness."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")
        if self.num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be positive, got {self.num_key_value_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if self.intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {self.intermediate_size}")
        if self.max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be positive, got {self.max_position_embeddings}")
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        if self.initializer_range <= 0:
            raise ValueError(f"initializer_range must be positive, got {self.initializer_range}")

        # Explicit head_dim validation: hidden_size must exactly equal num_attention_heads * head_dim
        expected_hidden_size = self.num_attention_heads * self.head_dim
        if self.hidden_size != expected_hidden_size:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must equal "
                f"num_attention_heads ({self.num_attention_heads}) * head_dim ({self.head_dim}) = {expected_hidden_size}"
            )

        # GQA grouping validation: query heads must be evenly divisible by key-value heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for Rotary Positional Embeddings (RoPE), got {self.head_dim}")

    @property
    def num_key_value_groups(self) -> int:
        """Returns the number of query heads assigned to each key/value head."""
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from a dictionary."""
        return cls(**data)

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """Preset configuration for tiny test model."""
        return cls(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=256,
            max_position_embeddings=512,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=True,
            initializer_range=0.02,
        )

    @classmethod
    def sproutko_130m(cls) -> "ModelConfig":
        """Preset configuration for SproutKO-130M (129,983,232 unique parameters)."""
        return cls(
            vocab_size=32_000,
            hidden_size=768,
            num_hidden_layers=16,
            num_attention_heads=12,
            num_key_value_heads=4,
            head_dim=64,
            intermediate_size=2176,
            max_position_embeddings=4096,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=True,
            initializer_range=0.02,
        )

    @classmethod
    def sproutko_300m(cls) -> "ModelConfig":
        """Preset configuration for SproutKO-300M model (~300M parameters)."""
        return cls(
            vocab_size=32_000,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=64,
            intermediate_size=2816,
            max_position_embeddings=4096,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=True,
            initializer_range=0.02,
        )

    @classmethod
    def sproutko_1b(cls) -> "ModelConfig":
        """Preset configuration for SproutKO-1B model (~1B parameters)."""
        return cls(
            vocab_size=32_000,
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
            intermediate_size=5632,
            max_position_embeddings=4096,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=True,
            initializer_range=0.02,
        )

    @classmethod
    def sproutko_7b(cls) -> "ModelConfig":
        """Preset configuration for SproutKO-7B model (~7B parameters)."""
        return cls(
            vocab_size=32_000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=11008,
            max_position_embeddings=4096,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=False,
            initializer_range=0.02,
        )

    @classmethod
    def sproutko_30b(cls) -> "ModelConfig":
        """Preset configuration for SproutKO-30B model (~30B parameters)."""
        return cls(
            vocab_size=32_000,
            hidden_size=6656,
            num_hidden_layers=60,
            num_attention_heads=52,
            num_key_value_heads=4,
            head_dim=128,
            intermediate_size=17920,
            max_position_embeddings=4096,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=False,
            initializer_range=0.02,
        )
