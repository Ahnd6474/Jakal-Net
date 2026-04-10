# Jakal-Net

Latent workspace multimodal model in PyTorch.

구현된 기본 구조는 아래 설계를 그대로 따른다.

- `Text Encoder -> Text nodes`
- `Image Encoder -> Image nodes`
- `T/I -> L collection`
- `discard T/I`
- `latent-only propagation with Softplus gating`
- `read back to text/image`
- `contrastive / matching / grounding losses`

현재 구현은 두 경로를 모두 지원한다.

- text encoder: 작은 Transformer encoder
- image encoder: 작은 ViT-style patch encoder
- readout: pooled latent residual injection
- COCO용 pretrained encoder: `distilbert-base-uncased` + `torchvision resnet18`

실행 예시:

```bash
PYTHONPATH=src python scripts/smoke_test.py
```

COCO 학습:

COCO 2017 기준으로 아래 두 경로가 필요하다.

- 이미지 디렉터리 예시: `train2017/`
- 캡션 주석 파일 예시: `annotations/captions_train2017.json`

pretrained encoder를 처음 실행할 때는 text 가중치는 Hugging Face에서, image 가중치는 torchvision 경로에서 내려받는다.

```bash
PYTHONPATH=src ./.venv/bin/python scripts/train_coco.py \
  --train-images-dir /path/to/coco/train2017 \
  --train-annotations /path/to/coco/annotations/captions_train2017.json \
  --batch-size 16 \
  --epochs 1
```
