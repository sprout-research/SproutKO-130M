"""Load the unchanged embedded Rust BPE state for inference."""
import json
from typing import Any, Dict


def load_rust_backend(state: Dict[str, Any]):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies with pip install -e .") from exc
    return Tokenizer.from_str(json.dumps(state, ensure_ascii=False, separators=(",", ":")))
