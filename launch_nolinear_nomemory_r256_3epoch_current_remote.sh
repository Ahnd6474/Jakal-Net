#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Jakal-Net
mkdir -p artifacts/launch_logs
pkill -f train_causal_memory_lm.py || true
nohup env \
  PYTHONPATH=src \
  JAKAL_NET_ENABLE_EXPERIMENTAL_FUSED_TRAINING=1 \
  JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_FASTPATH=1 \
  JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_LINEAR_FASTPATH=1 \
  /venv/main/bin/python scripts/train_causal_memory_lm.py \
  --device cuda \
  --precision bf16 \
  --model-kind causal_memory \
  --implementation native \
  --enable-fused-training \
  --disable-memory \
  --disable-memory-feed-forward-layers \
  --direction-only-values \
  --feed-forward-kind linear \
  --seq-len 512 \
  --dim 384 \
  --s-layers 6 \
  --prediction-layers 3 \
  --s-window 512 \
  --prediction-window 512 \
  --memory-slots 384 96 24 \
  --memory-update-intervals 1 2 4 \
  --pairwise-kind low_rank_bilinear \
  --route-kind low_rank_bilinear \
  --pairwise-rank 256 \
  --route-rank 256 \
  --pairwise-heads 8 \
  --route-heads 8 \
  --pairwise-anchor-heads 0 \
  --route-anchor-heads 0 \
  --batch-size 384 \
  --grad-accum-steps 1 \
  --optimizer adamw_fused \
  --learning-rate 5e-4 \
  --linear-learning-rate 5e-4 \
  --weight-decay 0.01 \
  --warmup-start-lr 1e-4 \
  --warmup-steps 500 \
  --linear-warmup-start-lr 1e-4 \
  --linear-warmup-steps 500 \
  --lr-min-ratio 0.2 \
  --linear-lr-min-ratio 0.2 \
  --linear-lr-delay-steps 0 \
  --lr-decay-start-step 4000 \
  --lr-decay-steps 0 \
  --eval-interval 500 \
  --prefetch-batches 100 \
  --prefetch-pin-memory \
  --prebuild-train-batches \
  --prebuild-pin-memory \
  --prebuild-workers 8 \
  --prebuild-worker-threads 2 \
  --curriculum-stage1-span 1 \
  --curriculum-stage2-span 1 \
  --curriculum-stage3-span 1 \
  --tensorboard \
  --pretokenized-dir artifacts/wiki2m_hf16k/pretokenized_shards_50k \
  --epochs 1.0 \
  --run-name wiki2m-nomemory-linear-b384-r256-h8-softsign-prebuild8x2-span1-1epoch-eval500-adamwfused-lr50e4-sharedlr-floor1e4-currentcode-20260503 \
  > artifacts/launch_logs/wiki2m-nomemory-linear-b384-r256-h8-softsign-prebuild8x2-span1-1epoch-eval500-adamwfused-lr50e4-sharedlr-floor1e4-currentcode-20260503.log 2>&1 < /dev/null &
echo pid=$!
echo log=artifacts/launch_logs/wiki2m-nomemory-linear-b384-r256-h8-softsign-prebuild8x2-span1-1epoch-eval500-adamwfused-lr50e4-sharedlr-floor1e4-currentcode-20260503.log
