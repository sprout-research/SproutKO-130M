"""Tokenizer configuration for SproutKO models."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Tuple

PLACEHOLDER_PREFIXES: Tuple[str, ...] = ("<reserved_", "<control_", "<unused_")


def is_placeholder_token(token: str) -> bool:
    """Returns True for reserved, control, or unused placeholder tokens."""
    return token.startswith(PLACEHOLDER_PREFIXES)


@dataclass
class TokenizerConfig:
    """Configuration class for SproutKOTokenizer.

    Attributes:
        vocab_size: Target vocabulary size (default: 32,000).
        pad_token: String for padding token (fixed ID: 0).
        bos_token: String for beginning-of-sequence token (fixed ID: 1).
        eos_token: String for end-of-sequence token (fixed ID: 2).
        unk_token: String for unknown token fallback indicator (fixed ID: 3).
        byte_fallback: Whether to reserve 256 byte tokens <0x00>..<0xFF> for lossless fallback.
        space_prefix: Surface character used to mark whitespace boundaries (default: '▁' U+2581).
        normalization: Unicode normalization form ('nfc', 'none').
        min_frequency: Minimum pair frequency required during BPE merge learning.
    """

    vocab_size: int = 32_000
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    unk_token: str = "<unk>"
    byte_fallback: bool = True
    space_prefix: str = "\u2581"  # '▁'
    normalization: str = "nfc"
    min_frequency: int = 2
    base_alphabet_policy: str = "frequency_covered"  # 'observed', 'frequency_covered', 'full_hangul'
    base_hangul_count: int = 2350
    reserved_control_tokens: int = 0
    backend: str = "rust_bpe"
    limit_alphabet: int = 8_000
    max_token_length: int = 32

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.min_frequency <= 0:
            raise ValueError(f"min_frequency must be positive, got {self.min_frequency}")
        if self.base_hangul_count < 0:
            raise ValueError(f"base_hangul_count must be non-negative, got {self.base_hangul_count}")
        if self.reserved_control_tokens < 0:
            raise ValueError(f"reserved_control_tokens must be non-negative, got {self.reserved_control_tokens}")
        if self.backend not in {"rust_bpe", "python_bpe"}:
            raise ValueError("backend must be 'rust_bpe' or 'python_bpe'")
        if self.limit_alphabet <= 0:
            raise ValueError(f"limit_alphabet must be positive, got {self.limit_alphabet}")
        if self.max_token_length <= 0:
            raise ValueError(f"max_token_length must be positive, got {self.max_token_length}")
        if self.normalization.lower() not in {"nfc", "none"}:
            raise ValueError(f"normalization must be 'nfc' or 'none', got {self.normalization!r}")
        if len(self.space_prefix) != 1:
            raise ValueError("space_prefix must be exactly one Unicode character")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("pad, bos, eos, and unk tokens must be distinct")

    @property
    def special_tokens(self) -> List[str]:
        """Returns ordered list of standard special tokens."""
        return [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def sproutko_32k(cls) -> "TokenizerConfig":
        """Return the immutable production specification for SproutKO-Tokenizer-32K-v1."""
        return cls(
            vocab_size=32_000,
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            byte_fallback=True,
            space_prefix="\u2581",
            normalization="nfc",
            min_frequency=2,
            base_alphabet_policy="frequency_covered",
            base_hangul_count=2350,
            reserved_control_tokens=0,
            backend="rust_bpe",
            limit_alphabet=8_000,
            max_token_length=32,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenizerConfig":
        """Create configuration from a dictionary and reject unknown semantic fields."""
        valid = {item.name for item in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown tokenizer configuration keys: {sorted(unknown)}")
        return cls(**data)
