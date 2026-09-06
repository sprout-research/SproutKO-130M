"""SproutKO tokenizer API backed by production Rust BPE or legacy pure Python BPE."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from sproutko.tokenizer.bpe import bpe_encode_word, token_to_byte
from sproutko.tokenizer.config import TokenizerConfig, is_placeholder_token
from sproutko.tokenizer.normalization import prepare_tokenizer_text, restore_tokenizer_text
from sproutko.tokenizer.pretokenizer import pretokenize
from sproutko.tokenizer.rust_backend import load_rust_backend
from sproutko.tokenizer.serialization import (
    load_tokenizer_from_file,
    save_tokenizer_to_file,
    serialize_tokenizer_data,
    validate_tokenizer_components,
)


class SproutKOTokenizer:
    """Stable SproutKO tokenizer API with a versioned, self-contained artifact."""

    def __init__(
        self,
        config: TokenizerConfig,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        backend_state: Optional[Dict[str, Any]] = None,
        backend: Optional[Any] = None,
    ) -> None:
        validate_tokenizer_components(config, vocab, merges, backend_state)
        self.config = config
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.backend_state = backend_state
        self._backend = backend
        if config.backend == "rust_bpe" and self._backend is None:
            assert backend_state is not None
            self._backend = load_rust_backend(backend_state)
        if config.backend == "rust_bpe":
            backend_vocab = self._backend.get_vocab(with_added_tokens=True)
            if backend_vocab != self.vocab:
                raise ValueError("Embedded Rust backend vocabulary does not match the artifact envelope")

        self.vocab_set: Set[str] = set(self.vocab)
        self.inverse_vocab: Dict[int, str] = {idx: tok for tok, idx in self.vocab.items()}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {pair: index for index, pair in enumerate(self.merges)}
        self._bpe_cache: Dict[str, Tuple[str, ...]] = {}
        self._bpe_cache_limit = 131_072

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.config.pad_token]

    @property
    def bos_token_id(self) -> int:
        return self.vocab[self.config.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.config.eos_token]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.config.unk_token]

    def semantic_payload(self) -> Dict[str, Any]:
        """Return the full canonical payload used for semantic fingerprinting."""
        return serialize_tokenizer_data(
            self.config,
            self.vocab,
            self.merges,
            self.backend_state,
        )

    def _bpe_chunk(self, chunk: str) -> List[str]:
        cached = self._bpe_cache.get(chunk)
        if cached is not None:
            return list(cached)
        subwords = bpe_encode_word(
            chunk,
            self.bpe_ranks,
            self.vocab_set,
            byte_fallback=self.config.byte_fallback,
        )
        if len(self._bpe_cache) < self._bpe_cache_limit:
            self._bpe_cache[chunk] = tuple(subwords)
        return subwords

    def _prepared(self, text: str) -> str:
        return prepare_tokenizer_text(
            text,
            form=self.config.normalization,
            space_prefix=self.config.space_prefix,
            special_tokens=self.config.special_tokens,
        )

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        prepared = self._prepared(text)
        if self.config.backend == "rust_bpe":
            return list(self._backend.encode(prepared, add_special_tokens=False).tokens)
        tokens: List[str] = []
        for chunk in pretokenize(prepared, space_prefix=self.config.space_prefix):
            tokens.extend(self._bpe_chunk(chunk))
        return tokens

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        token_ids: List[int] = []
        if add_bos:
            token_ids.append(self.bos_token_id)
        if text:
            if self.config.backend == "rust_bpe":
                token_ids.extend(self._backend.encode(self._prepared(text), add_special_tokens=False).ids)
            else:
                token_ids.extend(self.vocab.get(token, self.unk_token_id) for token in self.tokenize(text))
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        if not token_ids:
            return ""
        special_set = set(self.config.special_tokens)
        if self.config.backend == "rust_bpe":
            filtered = []
            for token_id in token_ids:
                token = self.inverse_vocab.get(token_id, self.config.unk_token)
                if skip_special_tokens and (token in special_set or is_placeholder_token(token)):
                    continue
                filtered.append(token_id if token_id in self.inverse_vocab else self.unk_token_id)
            decoded = self._backend.decode(filtered, skip_special_tokens=False)
            return restore_tokenizer_text(
                decoded,
                space_prefix=self.config.space_prefix,
                special_tokens=self.config.special_tokens,
            )

        segments: List[str] = []
        byte_buffer = bytearray()
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, self.config.unk_token)
            if skip_special_tokens and (token in special_set or is_placeholder_token(token)):
                continue
            byte_value = token_to_byte(token)
            if byte_value is not None:
                byte_buffer.append(byte_value)
                continue
            if byte_buffer:
                segments.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()
            segments.append(token)
        if byte_buffer:
            segments.append(byte_buffer.decode("utf-8", errors="replace"))
        restored_spaces = "".join(segments).replace(self.config.space_prefix, " ")
        return restore_tokenizer_text(
            restored_spaces,
            space_prefix=self.config.space_prefix,
            special_tokens=self.config.special_tokens,
        )

    def save(self, file_path: Union[str, Path]) -> None:
        save_tokenizer_to_file(
            file_path,
            self.config,
            self.vocab,
            self.merges,
            self.backend_state,
        )

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "SproutKOTokenizer":
        config, vocab, merges, backend_state = load_tokenizer_from_file(file_path)
        return cls(config=config, vocab=vocab, merges=merges, backend_state=backend_state)

    @classmethod
    def from_pretrained(
        cls,
        source: Union[str, Path],
        *,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> "SproutKOTokenizer":
        from sproutko.pretrained import load_tokenizer, pretrained_hub_kwargs

        return load_tokenizer(source, **pretrained_hub_kwargs(revision, token, cache_dir))

