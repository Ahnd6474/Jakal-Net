#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/Jakal-Net-docrun"
PYTHON_EXE="/workspace/Jakal-Net-dialogue/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts"

BUILD_ROOT="$REPO_ROOT/artifacts/data_full_v1"
RUN_NAME="cmem-full-v1-d640-s3-m256-64-16-p2-seq512-bs4-ga4"

cd "$REPO_ROOT"

rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"

echo "[build] plain dialogue half"
"$PYTHON_EXE" scripts/build_plain_dialogue_corpus.py \
  --output "$BUILD_ROOT/plain_dialogue_half.jsonl" \
  --source-label ultrachat \
  --max-records-per-source 100000 \
  --skip-failed-sources

echo "[build] pure code half"
"$PYTHON_EXE" scripts/build_segmented_dialogue_corpus.py \
  --output "$BUILD_ROOT/pure_code_half.jsonl" \
  --source-label code \
  --max-records-per-source 500000 \
  --skip-failed-sources

echo "[build] pure math half"
"$PYTHON_EXE" scripts/build_segmented_dialogue_corpus.py \
  --output "$BUILD_ROOT/pure_math_half.jsonl" \
  --source-label math \
  --max-records-per-source 500000 \
  --skip-failed-sources

echo "[build] pure wiki half"
"$PYTHON_EXE" scripts/build_segmented_dialogue_corpus.py \
  --output "$BUILD_ROOT/pure_wiki_half.jsonl" \
  --source-label wiki \
  --max-records-per-source 500000 \
  --skip-failed-sources

echo "[build] pure pubmed half"
"$PYTHON_EXE" scripts/build_segmented_dialogue_corpus.py \
  --output "$BUILD_ROOT/pure_pubmed_half.jsonl" \
  --source-label pubmed \
  --max-records-per-source 300000 \
  --skip-failed-sources

echo "[build] pure arxiv all"
"$PYTHON_EXE" scripts/build_segmented_dialogue_corpus.py \
  --output "$BUILD_ROOT/pure_arxiv_all.jsonl" \
  --source-label arxiv \
  --max-records-per-source 1000000 \
  --skip-failed-sources

echo "[build] mixed dialogue all"
"$PYTHON_EXE" scripts/build_segmented_dialogue_corpus.py \
  --output "$BUILD_ROOT/mixed_dialogue_all.jsonl" \
  --source-label python_code \
  --source-label codeact \
  --source-label metamath \
  --max-records-per-source 1000000 \
  --skip-failed-sources

echo "[build] math qa all"
"$PYTHON_EXE" scripts/build_math_qa_corpus.py \
  --output "$BUILD_ROOT/math_qa_all.jsonl" \
  --max-records-per-source 1000000 \
  --skip-failed-sources

echo "[build] reasoning all"
"$PYTHON_EXE" scripts/build_reasoning_dialogue_corpus.py \
  --output "$BUILD_ROOT/reasoning_all.jsonl" \
  --max-records-per-source 1000000 \
  --skip-failed-sources

echo "[build] line counts"
wc -l "$BUILD_ROOT"/*.jsonl

echo "[train] starting"
"$PYTHON_EXE" scripts/train_causal_memory_lm.py \
  --jsonl-source "$BUILD_ROOT/plain_dialogue_half.jsonl" \
  --jsonl-source "$BUILD_ROOT/pure_code_half.jsonl" \
  --jsonl-source "$BUILD_ROOT/pure_math_half.jsonl" \
  --jsonl-source "$BUILD_ROOT/pure_wiki_half.jsonl" \
  --jsonl-source "$BUILD_ROOT/pure_pubmed_half.jsonl" \
  --jsonl-source "$BUILD_ROOT/pure_arxiv_all.jsonl" \
  --jsonl-source "$BUILD_ROOT/mixed_dialogue_all.jsonl" \
  --jsonl-source "$BUILD_ROOT/math_qa_all.jsonl" \
  --jsonl-source "$BUILD_ROOT/reasoning_all.jsonl" \
  --tokenizer byte_bpe \
  --subword-vocab-size 16384 \
  --pretokenize-workers 8 \
  --device auto \
  --precision bf16 \
  --seq-len 512 \
  --dim 640 \
  --s-layers 3 \
  --memory-slots 256 64 16 \
  --prediction-layers 2 \
  --s-window 256 \
  --prediction-window 64 \
  --memory-topk 16 \
  --scan-backend native \
  --pairwise-kind low_rank_bilinear \
  --route-kind low_rank_bilinear \
  --pairwise-rank 40 \
  --route-rank 40 \
  --implementation native \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --warmup-steps 200 \
  --lr-min-ratio 0.1 \
  --epochs 1.0 \
  --train-fraction 0.9 \
  --grad-clip 1.0 \
  --eval-interval 100 \
  --eval-documents 12 \
  --curriculum-stage1-ratio 0.2 \
  --curriculum-stage2-ratio 0.6 \
  --curriculum-stage2-span 4 \
  --curriculum-stage3-span 8 \
  --output-root artifacts/training_runs \
  --run-name "$RUN_NAME" \
  --tensorboard
