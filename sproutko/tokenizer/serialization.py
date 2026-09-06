"""Versioned, validated, and atomic tokenizer artifact serialization."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sproutko.tokenizer.bpe import byte_to_token
from sproutko.tokenizer.config import TokenizerConfig

CURRENT_FORMAT_VERSION = "2.0"


def validate_tokenizer_components(
    config: TokenizerConfig,
    vocab: Dict[str, int],
    merges: List[Tuple[str, str]],
    backend_state: Optional[Dict[str, Any]],
) -> None:
    """Validate every invariant that affects token IDs or encoding semantics."""
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Tokenizer vocab must be a non-empty object")
    if any(not isinstance(token, str) or not isinstance(token_id, int) for token, token_id in vocab.items()):
        raise ValueError("Tokenizer vocab must map string tokens to integer IDs")
    ids = list(vocab.values())
    if len(set(ids)) != len(ids):
        raise ValueError("Tokenizer vocab contains duplicate token IDs")
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("Tokenizer token IDs must be contiguous from 0 to vocab_size-1")
    if len(vocab) != config.vocab_size:
        raise ValueError(f"Tokenizer config vocab_size={config.vocab_size} does not match artifact vocab={len(vocab)}")
    for expected_id, token in enumerate(config.special_tokens):
        if vocab.get(token) != expected_id:
            raise ValueError(f"Special token {token!r} must have fixed ID {expected_id}")

    if config.byte_fallback:
        for value in range(256):
            token = byte_to_token(value)
            if token not in vocab:
                raise ValueError(f"Tokenizer is missing byte fallback token {token}")

    seen_merges = set()
    for pair in merges:
        if not isinstance(pair, tuple) or len(pair) != 2 or not all(isinstance(item, str) for item in pair):
            raise ValueError(f"Invalid BPE merge entry: {pair!r}")
        if pair in seen_merges:
            raise ValueError(f"Duplicate BPE merge entry: {pair!r}")
        seen_merges.add(pair)
        if pair[0] not in vocab or pair[1] not in vocab:
            raise ValueError(f"BPE merge references a token outside the vocabulary: {pair!r}")
        if pair[0] + pair[1] not in vocab:
            raise ValueError(f"BPE merge product is missing from the vocabulary: {pair!r}")

    if config.backend == "rust_bpe" and not isinstance(backend_state, dict):
        raise ValueError("Rust BPE artifacts must embed backend_state")
    if config.backend == "rust_bpe" and isinstance(backend_state, dict):
        model_state = backend_state.get("model")
        pretokenizer_state = backend_state.get("pre_tokenizer")
        if not isinstance(model_state, dict) or model_state.get("type") != "BPE":
            raise ValueError("Rust backend_state must contain a BPE model")
        if bool(model_state.get("byte_fallback")) != config.byte_fallback:
            raise ValueError("Rust backend byte_fallback does not match tokenizer config")
        if model_state.get("unk_token") != config.unk_token:
            raise ValueError("Rust backend unk_token does not match tokenizer config")
        if not isinstance(pretokenizer_state, dict) or pretokenizer_state.get("type") != "Metaspace":
            raise ValueError("Rust backend_state must use the Metaspace pre-tokenizer")
        if pretokenizer_state.get("replacement") != config.space_prefix:
            raise ValueError("Rust backend Metaspace replacement does not match tokenizer config")
        if pretokenizer_state.get("prepend_scheme") != "never":
            raise ValueError("Rust backend Metaspace must use prepend_scheme='never'")
    if config.backend == "python_bpe" and backend_state is not None:
        raise ValueError("Python BPE artifacts must not contain backend_state")


def serialize_tokenizer_data(
    config: TokenizerConfig,
    vocab: Dict[str, int],
    merges: List[Tuple[str, str]],
    backend_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prepare the canonical tokenizer artifact payload."""
    validate_tokenizer_components(config, vocab, merges, backend_state)
    return {
        "format_version": CURRENT_FORMAT_VERSION,
        "backend": config.backend,
        "config": config.to_dict(),
        "vocab": vocab,
        "merges": [list(pair) for pair in merges],
        "backend_state": backend_state,
    }


def save_tokenizer_to_file(
    file_path: Union[str, Path],
    config: TokenizerConfig,
    vocab: Dict[str, int],
    merges: List[Tuple[str, str]],
    backend_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically save a validated tokenizer artifact as canonical compact JSON."""
    data = serialize_tokenizer_data(config, vocab, merges, backend_state)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(data, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def load_tokenizer_from_file(
    file_path: Union[str, Path],
) -> Tuple[TokenizerConfig, Dict[str, int], List[Tuple[str, str]], Optional[Dict[str, Any]]]:
    """Load and fully validate a tokenizer artifact."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Corrupted tokenizer file: expected JSON object at root, got {type(data)}")
    format_version = data.get("format_version")
    if format_version != CURRENT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported tokenizer format_version: {format_version!r}, expected {CURRENT_FORMAT_VERSION!r}"
        )
    for required_key in ("backend", "config", "vocab", "merges", "backend_state"):
        if required_key not in data:
            raise ValueError(f"Missing required key in tokenizer JSON: {required_key!r}")
    if not isinstance(data["config"], dict):
        raise ValueError("Tokenizer config must be a JSON object")
    config = TokenizerConfig.from_dict(data["config"])
    if data["backend"] != config.backend:
        raise ValueError("Tokenizer backend field does not match config.backend")
    if not isinstance(data["merges"], list):
        raise ValueError("Tokenizer merges must be a JSON array")
    merges: List[Tuple[str, str]] = []
    for pair in data["merges"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Invalid BPE merge entry: {pair!r}")
        merges.append((pair[0], pair[1]))
    vocab = data["vocab"]
    backend_state = data["backend_state"]
    validate_tokenizer_components(config, vocab, merges, backend_state)
    return config, vocab, merges, backend_state
