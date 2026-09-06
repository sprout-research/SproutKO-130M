"""Generate text from a SproutKO inference bundle using KV-cache decoding."""

import argparse
import sys
from pathlib import Path

import torch

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from sproutko.generation import generate
from sproutko.model import SproutKOForCausalLM
from sproutko.tokenizer.tokenizer import SproutKOTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a SproutKO causal LM checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Hub repo id or local safetensors export folder",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer JSON override",
    )
    parser.add_argument("--prompt", type=str, default="한국어는")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for sampling.")
    args = parser.parse_args()

    if Path(args.checkpoint).suffix.lower() in {".pt", ".pth", ".ckpt"}:
        parser.error("Use a safetensors export folder or Hub repo id; training checkpoints are unsupported.")
    model = SproutKOForCausalLM.from_pretrained(args.checkpoint)
    tokenizer = (
        SproutKOTokenizer.load(args.tokenizer)
        if args.tokenizer
        else SproutKOTokenizer.from_pretrained(args.checkpoint)
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    model.to(device)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt, add_bos=False, add_eos=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
