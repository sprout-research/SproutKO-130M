"""Exercise bundle loading without a trainer or external downloads."""
import json
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import ByteFallback, Fuse, Sequence
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from sproutko import ModelConfig, SproutKOForCausalLM, SproutKOTokenizer, TokenizerConfig


def test_bundle_load_preserves_logits_and_tokenizer(tmp_path):
    cfg = TokenizerConfig(vocab_size=260)
    vocab = {t: i for i, t in enumerate(cfg.special_tokens)}
    vocab.update({f"<0x{i:02X}>": i + 4 for i in range(256)})
    backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token=cfg.unk_token, byte_fallback=True))
    backend.pre_tokenizer = Metaspace(replacement=cfg.space_prefix, prepend_scheme="never")
    backend.decoder = Sequence([ByteFallback(), Fuse(), MetaspaceDecoder(replacement=cfg.space_prefix, prepend_scheme="never")])
    tok = SproutKOTokenizer(cfg, vocab, [], backend_state=json.loads(backend.to_str()))
    tok.save(tmp_path / "tokenizer.json")
    config = ModelConfig(vocab_size=260, hidden_size=32, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         intermediate_size=64, max_position_embeddings=64)
    model = SproutKOForCausalLM(config).eval()
    save_file({k: v.detach().contiguous().clone() for k, v in model.state_dict().items()}, str(tmp_path / "model.safetensors"))
    raw = config.to_dict()
    raw["tokenizer_file"] = "tokenizer.json"
    (tmp_path / "config.json").write_text(json.dumps(raw))
    loaded = SproutKOForCausalLM.from_pretrained(tmp_path)
    loaded_tok = SproutKOTokenizer.from_pretrained(tmp_path)
    assert not loaded.training
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    for text in ["한국어", "  공백  유지\n탭\t", "Hello 123 🌱", "한글", ""]:
        assert loaded_tok.encode(text) == tok.encode(text)
        assert loaded_tok.decode(tok.encode(text)) == tok.decode(tok.encode(text))
    ids = torch.tensor([tok.encode("한국어")])
    with torch.no_grad():
        torch.testing.assert_close(model(ids), loaded(ids), rtol=0, atol=0)


def test_missing_bundle_config_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        SproutKOForCausalLM.from_pretrained(tmp_path)
