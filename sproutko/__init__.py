"""SproutKO inference API with lazy PyTorch imports."""
from importlib import import_module

from sproutko.config import ModelConfig
from sproutko.tokenizer import SproutKOTokenizer, TokenizerConfig

__version__ = "1.0.0"
_LAZY = {
    "SproutKOForCausalLM": ("sproutko.model", "SproutKOForCausalLM"),
    "generate": ("sproutko.generation", "generate"),
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module 'sproutko' has no attribute {name!r}")
    module, attribute = _LAZY[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


__all__ = ["ModelConfig", "TokenizerConfig", "SproutKOTokenizer", *_LAZY]
