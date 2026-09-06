"""Script-aware pre-tokenization with whitespace preservation."""

import re
from typing import Dict, List

HANGUL_CLASS = r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]"
LATIN_CLASS = r"[A-Za-z]"
DIGIT_CLASS = r"[0-9]"
# CJK Unified Ideographs + extension A + compatibility + Hiragana + Katakana
CJK_CLASS = r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff]"

_PATTERN_CACHE: Dict[str, re.Pattern] = {}


def _compile_token_pattern(space_prefix: str) -> re.Pattern:
    """Compiles (and caches) the script-aware pretokenizer pattern for a space marker."""
    cached = _PATTERN_CACHE.get(space_prefix)
    if cached is not None:
        return cached

    prefix = re.escape(space_prefix)
    pattern = re.compile(
        rf"{prefix}*{HANGUL_CLASS}+"
        rf"|{prefix}*{LATIN_CLASS}+"
        rf"|{prefix}*{DIGIT_CLASS}+"
        rf"|{prefix}*{CJK_CLASS}+"
        rf"|\n+"
        rf"|\t+"
        rf"|{prefix}+"
        rf"|[^{prefix}\s]"
    )
    _PATTERN_CACHE[space_prefix] = pattern
    return pattern


# Default pattern used when the space prefix is the SentencePiece marker.
PRETOKEN_PATTERN = _compile_token_pattern("\u2581")


def pretokenize(text: str, space_prefix: str = "\u2581") -> List[str]:
    """Splits text into script-aware pre-token segments while preserving whitespace.

    Spaces preceding tokens are converted into space_prefix ('▁').

    Args:
        text: Normalized input string.
        space_prefix: Surface marker character for whitespace (default: '▁').

    Returns:
        List of pre-tokenized chunks.
    """
    if not text:
        return []

    # Leave newlines and tabs intact so they retain separate token identities.
    text_with_markers = text.replace(" ", space_prefix)
    token_pattern = _compile_token_pattern(space_prefix)
    tokens = token_pattern.findall(text_with_markers)

    if "".join(tokens) != text_with_markers:
        tokens = []
        index = 0
        length = len(text_with_markers)
        while index < length:
            match = token_pattern.match(text_with_markers, index)
            if match:
                tokens.append(match.group(0))
                index = match.end()
            else:
                tokens.append(text_with_markers[index])
                index += 1

    return tokens


def restore_pretokenized_text(tokens: List[str], space_prefix: str = "\u2581") -> str:
    """Restores original text from pre-tokenized chunks."""
    joined = "".join(tokens)
    return joined.replace(space_prefix, " ")
