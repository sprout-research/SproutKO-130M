# Release verification

- Original and inference runtime match on 9 tokenizer samples (Korean, English, numbers, whitespace, emoji, decomposed Hangul, special-token text, empty input, and multilingual text).
- Actual 130M weights: CPU forward logits SHA-256 and 12-token greedy output match exactly in the same environment.
- Model weights, config, and tokenizer file bytes are unchanged; checksums are in SHA256SUMS.
- Unit tests: 31 passed (generation, KV cache, config, and local safetensors/tokenizer bundle loading).
- Source distribution and wheel built successfully. Wheel contains no training/corpus modules or weights.
- Installed wheel imported and CPU CLI inference ran successfully outside both source checkouts.
- GPU inference and live Hub download were not tested.
- GitHub remote creation/publication and model Hub publication are pending.
