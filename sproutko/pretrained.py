"""Load SproutKO inference weights from a local export folder or the Hugging Face Hub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from sproutko.config import ModelConfig

MODEL_CONFIG_KEYS = tuple(ModelConfig().to_dict().keys())
DEFAULT_WEIGHTS_FILE = "model.safetensors"
DEFAULT_CONFIG_FILE = "config.json"

Source = Union[str, Path]


def _hub_install_hint() -> str:
    return "Install with: pip install sproutko"


def _require_safetensors_load():
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "Loading safetensors weights requires the safetensors package. " + _hub_install_hint()
        ) from exc
    return load_file


def _hf_hub_download(repo_id: str, filename: str, **hub_kwargs: Any) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Loading from the Hugging Face Hub requires huggingface_hub. " + _hub_install_hint()) from exc
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, **hub_kwargs))


def resolve_file(source: Source, filename: str, **hub_kwargs: Any) -> Path:
    """Resolve a bundle file from a local directory or the Hub."""
    path = Path(source)
    if path.is_dir():
        candidate = path / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"{filename} not found in {path}")
        return candidate
    return _hf_hub_download(str(source), filename, **hub_kwargs)


def load_bundle_config(source: Source, **hub_kwargs: Any) -> Dict[str, Any]:
    config_path = resolve_file(source, DEFAULT_CONFIG_FILE, **hub_kwargs)
    return json.loads(config_path.read_text(encoding="utf-8"))


def model_config_from_bundle(raw: Dict[str, Any]) -> ModelConfig:
    """Keep only `ModelConfig` fields so extra Hub keys do not break `from_dict`."""
    missing = [key for key in MODEL_CONFIG_KEYS if key not in raw]
    if missing:
        raise KeyError(f"config.json missing ModelConfig keys: {missing}")
    keys = {key: raw[key] for key in MODEL_CONFIG_KEYS}
    return ModelConfig.from_dict(keys)


def resolve_tokenizer_filename(raw: Dict[str, Any], source: Source) -> str:
    name = raw.get("tokenizer_file")
    if isinstance(name, str) and name:
        return name
    path = Path(source)
    if path.is_dir():
        candidates = [
            item.name
            for item in path.glob("*.json")
            if item.name != DEFAULT_CONFIG_FILE and "manifest" not in item.name.lower()
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(f"No tokenizer JSON found in {path}")
        raise FileNotFoundError(f"Multiple tokenizer JSON files in {path}: {candidates}")
    raise FileNotFoundError("config.json has no tokenizer_file; cannot infer the tokenizer on the Hub")


def load_causal_lm(source: Source, **hub_kwargs: Any):
    """Load `SproutKOForCausalLM` from a local export folder or Hub repo id."""
    from sproutko.model.causal_lm import SproutKOForCausalLM

    raw = load_bundle_config(source, **hub_kwargs)
    config = model_config_from_bundle(raw)
    weights_file = raw.get("weights_file") or DEFAULT_WEIGHTS_FILE
    weights_path = resolve_file(source, str(weights_file), **hub_kwargs)
    load_file = _require_safetensors_load()
    state = load_file(str(weights_path))
    model = SproutKOForCausalLM(config)
    model.load_state_dict(state)
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.eval()
    return model


def load_tokenizer(source: Source, **hub_kwargs: Any):
    """Load `SproutKOTokenizer` from a local export folder or Hub repo id."""
    from sproutko.tokenizer.tokenizer import SproutKOTokenizer

    raw = load_bundle_config(source, **hub_kwargs)
    filename = resolve_tokenizer_filename(raw, source)
    path = resolve_file(source, filename, **hub_kwargs)
    return SproutKOTokenizer.load(path)


def pretrained_hub_kwargs(
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if revision is not None:
        kwargs["revision"] = revision
    if token is not None:
        kwargs["token"] = token
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return kwargs
