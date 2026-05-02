#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Jakal-Net
export PYTHONPATH=src:scripts
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-32}

PY=/venv/main/bin/python
DATA=artifacts/data_wiki_2m/wiki_2m.jsonl
ROOT=artifacts/wiki2m_hf16k
TOK_DIR=$ROOT/tokenizer
PRETOK=$ROOT/pretokenized/wiki2m_hf_bpe16k_seq512.pt
LOG_DIR=$ROOT/logs

mkdir -p "$ROOT" "$LOG_DIR" "$(dirname "$DATA")" "$(dirname "$PRETOK")"

if [ ! -f "$DATA" ]; then
  echo "stage=export_wikipedia_jsonl" | tee -a "$LOG_DIR/setup.log"
  "$PY" scripts/export_wikipedia_stream_jsonl.py \
    --output "$DATA" \
    --limit 2000000 \
    --hard-exit \
    > "$LOG_DIR/export_wikipedia_jsonl.log" 2>&1
else
  echo "stage=export_wikipedia_jsonl skip_existing" | tee -a "$LOG_DIR/setup.log"
fi

if [ ! -f "$TOK_DIR/tokenizer.json" ]; then
  echo "stage=tokenizer_train" | tee -a "$LOG_DIR/setup.log"
  "$PY" scripts/train_hf_fast_bpe_tokenizer.py \
    --input "$DATA" \
    --output-dir "$TOK_DIR" \
    --vocab-size 16384 \
    > "$LOG_DIR/tokenizer_train.log" 2>&1
else
  echo "stage=tokenizer_train skip_existing" | tee -a "$LOG_DIR/setup.log"
fi

if [ ! -f "$PRETOK" ]; then
  echo "stage=pretokenize" | tee -a "$LOG_DIR/setup.log"
  "$PY" scripts/build_pretokenized_cache.py \
    --jsonl-source "$DATA" \
    --tokenizer-dir "$TOK_DIR" \
    --output "$PRETOK" \
    --seq-len 512 \
    --workers 16 \
    > "$LOG_DIR/pretokenize.log" 2>&1
else
  echo "stage=pretokenize skip_existing" | tee -a "$LOG_DIR/setup.log"
fi

echo "stage=done" | tee -a "$LOG_DIR/setup.log"
