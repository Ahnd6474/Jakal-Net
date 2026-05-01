#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Jakal-Net-ffn2-lion-24e5b7c
mkdir -p artifacts/launch_logs
pkill -f wiki2m-nomemory-b384-r256-h4-softsign-prebuild8x2-span1-1epoch-eval500-lr14e4-warmup200-20260430 || true
export PYTHONPATH=src
export JAKAL_NET_ENABLE_EXPERIMENTAL_FUSED_TRAINING=1
export JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_FASTPATH=1
nohup /venv/main/bin/python scripts/train_causal_memory_lm.py --device cuda --precision bf16 --model-kind causal_memory --implementation native --enable-fused-training --disable-memory --disable-feed-forward-layers --disable-memory-feed-forward-layers --direction-only-values --seq-len 512 --dim 384 --s-layers 6 --prediction-layers 3 --s-window 512 --prediction-window 512 --memory-slots 384 96 24 --memory-update-intervals 1 2 4 --pairwise-kind low_rank_bilinear --route-kind low_rank_bilinear --pairwise-rank 256 --route-rank 256 --pairwise-heads 4 --route-heads 4 --pairwise-anchor-heads 0 --route-anchor-heads 0 --batch-size 384 --grad-accum-steps 1 --optimizer lion --learning-rate 2e-4 --weight-decay 0.01 --warmup-start-lr 1e-4 --warmup-steps 200 --prefetch-batches 100 --prefetch-pin-memory --prebuild-train-batches --prebuild-pin-memory --prebuild-workers 8 --prebuild-worker-threads 2 --curriculum-stage1-span 1 --curriculum-stage2-span 1 --curriculum-stage3-span 1 --tensorboard --pretokenized-dir artifacts/wiki2m_hf16k/pretokenized_shards_50k --epochs 1.0 --eval-interval 500 --run-name wiki2m-nomemory-b384-r256-h4-softsign-prebuild8x2-span1-1epoch-eval500-lr20e4-warmup200-20260430 > artifacts/launch_logs/wiki2m-nomemory-b384-r256-h4-softsign-prebuild8x2-span1-1epoch-eval500-lr20e4-warmup200-20260430.log 2>&1 < /dev/null &
echo pid=$!
echo log=artifacts/launch_logs/wiki2m-nomemory-b384-r256-h4-softsign-prebuild8x2-span1-1epoch-eval500-lr20e4-warmup200-20260430.log
