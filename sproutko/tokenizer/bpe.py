"""Byte-Pair Encoding (BPE) core algorithms and merge operations."""

import re
from typing import Dict, List, Optional, Set, Tuple

BYTE_TOKEN_PATTERN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def byte_to_token(b: int) -> str:
    """Formats an integer byte (0-255) into a byte token string, e.g. <0x4A>."""
    return f"<0x{b:02X}>"


def is_byte_token(token: str) -> bool:
    """Checks whether a token string is a byte fallback token (<0x00>..<0xFF>)."""
    return bool(BYTE_TOKEN_PATTERN.match(token))


def token_to_byte(token: str) -> Optional[int]:
    """Parses a byte token string into an integer byte, or None if not a byte token."""
    match = BYTE_TOKEN_PATTERN.match(token)
    if match:
        return int(match.group(1), 16)
    return None


def get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    """Returns the set of all adjacent symbol pairs in a tuple of symbols."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def bpe_encode_word(
    word: str,
    bpe_ranks: Dict[Tuple[str, str], int],
    vocab: Set[str],
    byte_fallback: bool = True,
) -> List[str]:
    """Encodes a single pre-token string into subword tokens using learned BPE merge ranks.

    If a character is not in the vocabulary, it is converted into UTF-8 byte tokens if byte_fallback=True.

    Args:
        word: Input word / pre-token string.
        bpe_ranks: Dictionary mapping (pair_0, pair_1) -> merge priority rank.
        vocab: Set of valid vocabulary tokens.
        byte_fallback: Whether to use byte tokens for unknown characters.

    Returns:
        List of subword token strings.
    """
    if not word:
        return []

    # If the whole word is already in vocab, return it directly
    if word in vocab:
        return [word]

    # Convert characters to initial symbol list
    symbols: List[str] = []
    for char in word:
        if char in vocab:
            symbols.append(char)
        elif byte_fallback:
            # Fallback to UTF-8 byte tokens
            for b in char.encode("utf-8"):
                symbols.append(byte_to_token(b))
        else:
            symbols.append("<unk>")

    if len(symbols) <= 1:
        return symbols

    # Iteratively apply BPE merges based on lowest rank
    while len(symbols) >= 2:
        pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        # Find the pair with the lowest rank (highest priority)
        min_pair = min(pairs, key=lambda p: bpe_ranks.get(p, float("inf")))

        if min_pair not in bpe_ranks:
            break

        first, second = min_pair
        new_symbols: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                new_symbols.append(first + second)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1

        symbols = new_symbols

    return symbols
