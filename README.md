# SproutKO-130M

[English](README.md) | [한국어](README.ko.md)

An **inference-only runtime** for SproutKO-130M, a Korean pretrained base language model.
Training code, data processing code, and training logs are not included.
The code, weights, and tokenizer are licensed under [Apache-2.0](LICENSE).

## Installation

Python 3.10 or later is required. For GPU inference, install the appropriate PyTorch version for your environment first.

```bash
git clone https://github.com/project-iconik/SproutKO-130M.git
cd SproutKO-130M
pip install .
```

## Text generation

The following command can be used once the weights are public on the model Hub.
If the Hub repository is private, you need access to it and must authenticate locally.

```bash
sproutko-generate --checkpoint project-iconik/SproutKO-130M --prompt "한국어는" --max-new-tokens 64 --temperature 0 --device cpu
```

A local bundle must contain `config.json`, `model.safetensors`, and the tokenizer JSON specified in the configuration.
In the prepared local project, these files are copied into `weights/`. They are not tracked by Git.

```bash
sproutko-generate --checkpoint ./weights --prompt "한국어는" --temperature 0
python scripts/generate.py --checkpoint ./weights --prompt "한국어는" --temperature 0
```

```python
import torch
from sproutko import SproutKOForCausalLM, SproutKOTokenizer, generate

source = "project-iconik/SproutKO-130M"  # Or a local weights directory
model = SproutKOForCausalLM.from_pretrained(source)
tokenizer = SproutKOTokenizer.from_pretrained(source)
ids = torch.tensor([tokenizer.encode("한국어는")], dtype=torch.long)
output = generate(model, ids, max_new_tokens=64, temperature=0,
                  eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0].tolist()))
```

Use weights and tokenizer files from the same release together.
The Hub loaders also support `revision`, `token`, and `cache_dir` arguments.
`transformers.AutoModelForCausalLM` and training `.pt` checkpoints are not supported.
Empty prompts are not supported. The combined prompt and output length must not exceed 4,096 tokens.

## Model

The model has 129,983,232 unique parameters, 16 layers, a hidden size of 768,
12 query heads, 4 KV heads, and a vocabulary of 32,000 tokens. It uses RoPE, GQA, RMSNorm, and SwiGLU.
The pretraining sequence length is 2,048. This is a base model without conversational instruction tuning.
Generated text may contain inaccurate or biased content.

## Verification

```bash
pip install -e ".[dev]"
pytest -q
python -m build
```

Checksums for the release files are in `release/SHA256SUMS`.
The complete weights bundle is distributed through the model Hub, and the inference code through GitHub.
