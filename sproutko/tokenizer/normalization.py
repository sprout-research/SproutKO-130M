"""Unicode normalization policy for Korean and multilingual text."""

import unicodedata
from typing import Optional, Sequence

TEXT_ESCAPE = "\U000f0000"


def normalize_text(text: str, form: str = "nfc") -> str:
    """Applies canonical Unicode normalization and line ending standardization.

    Preserves Korean internet slang (ㅋㅋㅋ, ㅠㅠ), compatibility jamo, whitespace,
    and emojis without destructive filtering.

    Args:
        text: Input raw string.
        form: Normalization form ('nfc' or 'none').

    Returns:
        Normalized string.
    """
    if not text:
        return ""

    # Standardize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Canonical NFC normalization
    if form.lower() == "nfc":
        text = unicodedata.normalize("NFC", text)
    elif form.lower() == "nfd":
        text = unicodedata.normalize("NFD", text)
    elif form.lower() != "none":
        raise ValueError(f"Unsupported normalization form: {form}")

    # Remove Byte Order Mark (BOM)
    text = text.replace("\ufeff", "")

    return text


def prepare_tokenizer_text(
    text: str,
    form: str = "nfc",
    space_prefix: str = "\u2581",
    special_tokens: Optional[Sequence[str]] = None,
) -> str:
    """Normalize text and reversibly escape literal tokenizer marker characters."""
    normalized = normalize_text(text, form=form)
    escaped = normalized.replace(TEXT_ESCAPE, TEXT_ESCAPE + TEXT_ESCAPE).replace(
        space_prefix,
        TEXT_ESCAPE + "M",
    )
    for index, token in enumerate(special_tokens or ()):
        escaped = escaped.replace(token, f"{TEXT_ESCAPE}S{index};")
    return escaped


def restore_tokenizer_text(
    text: str,
    space_prefix: str = "\u2581",
    special_tokens: Optional[Sequence[str]] = None,
) -> str:
    """Reverse :func:`prepare_tokenizer_text` without reserving an unrepresentable code point."""
    restored = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != TEXT_ESCAPE:
            restored.append(char)
            index += 1
            continue
        if index + 1 >= len(text):
            restored.append(TEXT_ESCAPE)
            index += 1
            continue
        marker = text[index + 1]
        if marker == TEXT_ESCAPE:
            restored.append(TEXT_ESCAPE)
            index += 2
        elif marker == "M":
            restored.append(space_prefix)
            index += 2
        elif marker == "S":
            end = text.find(";", index + 2)
            raw_index = text[index + 2 : end] if end >= 0 else ""
            if end >= 0 and raw_index.isdigit() and int(raw_index) < len(special_tokens or ()):
                restored.append((special_tokens or ())[int(raw_index)])
                index = end + 1
            else:
                restored.append(TEXT_ESCAPE)
                index += 1
        else:
            restored.append(TEXT_ESCAPE)
            index += 1
    return "".join(restored)
