# SproutKO-130M

한국어 사전학습 기본 모델 SproutKO-130M의 **추론 전용 런타임**입니다.
학습 코드, 데이터 처리 코드, 학습 로그는 포함하지 않습니다.
코드·가중치·토크나이저의 라이선스는 [Apache-2.0](LICENSE)입니다.

## 설치

Python 3.10 이상이 필요합니다. GPU 사용 시 환경에 맞는 PyTorch를 먼저 설치하세요.

```bash
git clone https://github.com/project-iconik/SproutKO-130M.git
cd SproutKO-130M
pip install .
```

## 텍스트 생성

가중치가 모델 Hub에 공개된 후 다음 명령을 사용할 수 있습니다.
비공개 상태라면 해당 Hub 저장소 접근 권한과 로컬 로그인이 필요합니다.

```bash
sproutko-generate --checkpoint project-iconik/SproutKO-130M --prompt "한국어는" --max-new-tokens 64 --temperature 0 --device cpu
```

로컬 번들에는 `config.json`, `model.safetensors`, 설정에서 지정한 토크나이저 JSON이 필요합니다.
로컬 준비 폴더에는 이 파일들이 `weights/`에 복사되어 있습니다. Git에는 포함되지 않습니다.

```bash
sproutko-generate --checkpoint ./weights --prompt "한국어는" --temperature 0
python scripts/generate.py --checkpoint ./weights --prompt "한국어는" --temperature 0
```

```python
import torch
from sproutko import SproutKOForCausalLM, SproutKOTokenizer, generate

source = "project-iconik/SproutKO-130M"  # 또는 로컬 weights 폴더
model = SproutKOForCausalLM.from_pretrained(source)
tokenizer = SproutKOTokenizer.from_pretrained(source)
ids = torch.tensor([tokenizer.encode("한국어는")], dtype=torch.long)
output = generate(model, ids, max_new_tokens=64, temperature=0,
                  eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0].tolist()))
```

가중치와 토크나이저는 동일한 릴리스의 파일을 함께 사용하세요.
Hub 로더는 `revision`, `token`, `cache_dir` 인자도 지원합니다.
`transformers.AutoModelForCausalLM` 및 학습용 `.pt` 체크포인트는 지원하지 않습니다.
빈 프롬프트는 지원하지 않습니다. 프롬프트와 출력 길이의 합은 최대 4,096 토큰입니다.

## 모델

129,983,232개 고유 파라미터, 16개 레이어, hidden size 768,
query/KV head 12/4, 32,000개 어휘, RoPE·GQA·RMSNorm·SwiGLU를 사용합니다.
사전학습 시퀀스 길이는 2,048입니다. 대화형 지시 학습을 거치지 않은 기본 모델입니다.
출력에 부정확하거나 편향된 내용이 포함될 수 있습니다.

## 검증

```bash
pip install -e ".[dev]"
pytest -q
python -m build
```

공개 파일의 체크섬은 `release/SHA256SUMS`에 있습니다.
가중치 번들 전체는 모델 Hub에, 추론 코드는 GitHub에 배포합니다.
