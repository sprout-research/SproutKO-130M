"""KV-Cache representation and management for autoregressive inference."""

from typing import List, Optional, Tuple, Union

import torch


class LayerKVCache:
    """KV cache storage for a single Transformer layer.

    Stores unrepeated Key and Value projections with RoPE applied to Keys.
    Optional capacity preallocation lets decode steps write in-place instead of
    concatenating a new tensor every token.

    Shapes:
        key:   (batch_size, num_kv_heads, cached_seq_len, head_dim)
        value: (batch_size, num_kv_heads, cached_seq_len, head_dim)
    """

    def __init__(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.shape != value.shape:
            raise ValueError(f"Key and Value cache shapes must match, got key {key.shape} and value {value.shape}")
        if key.ndim != 4:
            raise ValueError(f"KV cache tensors must be 4D [B, H_kv, T, D], got shape {key.shape}")
        self._key = key
        self._value = value
        self._seq_len = int(key.shape[2])

    @classmethod
    def with_capacity(cls, key: torch.Tensor, value: torch.Tensor, max_seq_len: int) -> "LayerKVCache":
        """Copies ``key``/``value`` into a buffer of length ``max_seq_len`` when larger."""
        cache = cls(key, value)
        cache._ensure_capacity(max_seq_len)
        return cache

    def _ensure_capacity(self, max_seq_len: int) -> None:
        if max_seq_len <= self._key.shape[2]:
            return
        batch, heads, _, dim = self._key.shape
        new_key = self._key.new_empty((batch, heads, max_seq_len, dim))
        new_value = self._value.new_empty((batch, heads, max_seq_len, dim))
        filled = self._seq_len
        new_key[:, :, :filled] = self._key[:, :, :filled]
        new_value[:, :, :filled] = self._value[:, :, :filled]
        self._key = new_key
        self._value = new_value

    def append(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Writes new K/V along the sequence axis, growing the buffer if needed."""
        if key_states.shape != value_states.shape:
            raise ValueError(f"key_states and value_states shape mismatch: {key_states.shape} vs {value_states.shape}")
        if key_states.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, H_kv, T, D], got shape {key_states.shape}")
        if (
            key_states.shape[0] != self.batch_size
            or key_states.shape[1] != self.num_kv_heads
            or key_states.shape[3] != self.head_dim
        ):
            raise ValueError(
                f"Append shape mismatch: cache {(self.batch_size, self.num_kv_heads, self.head_dim)} "
                f"vs incoming {(key_states.shape[0], key_states.shape[1], key_states.shape[3])}"
            )
        incoming = int(key_states.shape[2])
        needed = self._seq_len + incoming
        if needed > self._key.shape[2]:
            self._ensure_capacity(max(needed, self._key.shape[2] * 2))
        start = self._seq_len
        self._key[:, :, start:needed] = key_states
        self._value[:, :, start:needed] = value_states
        self._seq_len = needed

    @property
    def key(self) -> torch.Tensor:
        return self._key[:, :, : self._seq_len]

    @property
    def value(self) -> torch.Tensor:
        return self._value[:, :, : self._seq_len]

    @property
    def seq_len(self) -> int:
        """Returns sequence length currently held in this layer's cache."""
        return self._seq_len

    @property
    def batch_size(self) -> int:
        """Returns batch size of cached tensors."""
        return self._key.shape[0]

    @property
    def num_kv_heads(self) -> int:
        """Returns number of key/value heads."""
        return self._key.shape[1]

    @property
    def head_dim(self) -> int:
        """Returns head dimension."""
        return self._key.shape[3]


class KVCache:
    """Dynamic multi-layer Key-Value cache manager for SproutKO models.

    Maintains independent LayerKVCache objects for each Transformer block.
    Supports incremental concatenation during prefill and single/multi-token decode.
    """

    def __init__(self, num_layers: Optional[int] = None) -> None:
        self.layers: List[Optional[LayerKVCache]] = [None] * num_layers if num_layers is not None else []

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> Optional[LayerKVCache]:
        return self.layers[index]

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """Returns cached sequence length for the specified layer (or 0 if uninitialized)."""
        if layer_idx < len(self.layers) and self.layers[layer_idx] is not None:
            layer = self.layers[layer_idx]
            assert layer is not None
            return layer.seq_len
        return 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends new key and value projections to the cache at layer_idx.

        Args:
            key_states: Current layer projected/rotated Key tensor [B, num_kv_heads, current_seq_len, head_dim].
            value_states: Current layer projected Value tensor [B, num_kv_heads, current_seq_len, head_dim].
            layer_idx: Index of the layer being updated.

        Returns:
            Tuple of (full_keys, full_values) of shape [B, num_kv_heads, total_seq_len, head_dim].
        """
        if key_states.shape != value_states.shape:
            raise ValueError(f"key_states and value_states shape mismatch: {key_states.shape} vs {value_states.shape}")
        if key_states.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, H_kv, T, D], got shape {key_states.shape}")

        # Expand layer list if needed
        while len(self.layers) <= layer_idx:
            self.layers.append(None)

        past_layer = self.layers[layer_idx]

        if past_layer is None:
            # Prefill / first step
            self.layers[layer_idx] = LayerKVCache(key=key_states, value=value_states)
            return key_states, value_states
        else:
            # Validate batch size, head count, head dim compatibility
            if key_states.shape[0] != past_layer.batch_size:
                raise ValueError(
                    f"Batch size mismatch: past cache had {past_layer.batch_size}, "
                    f"current update has {key_states.shape[0]}"
                )
            if key_states.shape[1] != past_layer.num_kv_heads:
                raise ValueError(
                    f"num_kv_heads mismatch: past cache had {past_layer.num_kv_heads}, "
                    f"current update has {key_states.shape[1]}"
                )
            if key_states.shape[3] != past_layer.head_dim:
                raise ValueError(
                    f"head_dim mismatch: past cache had {past_layer.head_dim}, current update has {key_states.shape[3]}"
                )

            past_layer.append(key_states, value_states)
            return past_layer.key, past_layer.value

    def validate(
        self,
        expected_num_layers: int,
        expected_num_kv_heads: int,
        expected_head_dim: int,
    ) -> None:
        """Validates all layer caches for architectural consistency."""
        if len(self.layers) != expected_num_layers:
            raise ValueError(f"KVCache has {len(self.layers)} layers, expected {expected_num_layers}")

        ref_seq_len: Optional[int] = None
        for i, layer in enumerate(self.layers):
            if layer is None:
                raise ValueError(f"Layer {i} cache is None in initialized KVCache")

            if layer.num_kv_heads != expected_num_kv_heads:
                raise ValueError(f"Layer {i} has {layer.num_kv_heads} KV heads, expected {expected_num_kv_heads}")
            if layer.head_dim != expected_head_dim:
                raise ValueError(f"Layer {i} has head_dim {layer.head_dim}, expected {expected_head_dim}")

            if ref_seq_len is None:
                ref_seq_len = layer.seq_len
            elif layer.seq_len != ref_seq_len:
                raise ValueError(
                    f"Inconsistent sequence lengths across layers: layer 0 has {ref_seq_len}, "
                    f"layer {i} has {layer.seq_len}"
                )

    @classmethod
    def from_legacy_tuple(
        cls,
        past_key_values: Union["KVCache", Tuple[Union[LayerKVCache, Tuple[torch.Tensor, torch.Tensor]], ...]],
    ) -> "KVCache":
        """Converts tuple representations into a unified KVCache instance."""
        if isinstance(past_key_values, cls):
            return past_key_values

        cache = cls(num_layers=len(past_key_values))
        for i, item in enumerate(past_key_values):
            if isinstance(item, LayerKVCache):
                cache.layers[i] = item
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                cache.layers[i] = LayerKVCache(key=item[0], value=item[1])
            elif item is None:
                cache.layers[i] = None
            else:
                raise ValueError(f"Unrecognized layer cache type at index {i}: {type(item)}")
        return cache
