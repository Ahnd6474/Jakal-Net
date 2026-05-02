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
SPLIT_DIR=artifacts/data_wiki_2m/shards_50k
OUT_DIR=$ROOT/pretokenized_shards_50k
LOG_DIR=$ROOT/logs/sharded_pretokenize
LINES_PER_SHARD=${LINES_PER_SHARD:-50000}
JOBS=${PRETOK_JOBS:-8}
WORKERS_PER_SHARD=${PRETOK_WORKERS_PER_SHARD:-2}

mkdir -p "$SPLIT_DIR" "$OUT_DIR" "$LOG_DIR"

if [ ! -f "$TOK_DIR/tokenizer.json" ]; then
  echo "missing tokenizer: $TOK_DIR/tokenizer.json" >&2
  exit 1
fi

if ! find "$SPLIT_DIR" -maxdepth 1 -name 'wiki_2m_*.jsonl' -print -quit | grep -q .; then
  echo "stage=split_jsonl" | tee -a "$ROOT/logs/sharded_setup.log"
  split -d -a 2 -l "$LINES_PER_SHARD" --additional-suffix=.jsonl "$DATA" "$SPLIT_DIR/wiki_2m_"
else
  echo "stage=split_jsonl skip_existing" | tee -a "$ROOT/logs/sharded_setup.log"
fi

echo "stage=build_shards jobs=$JOBS workers_per_shard=$WORKERS_PER_SHARD" | tee -a "$ROOT/logs/sharded_setup.log"

find "$SPLIT_DIR" -maxdepth 1 -type f -name 'wiki_2m_*.jsonl' | sort | \
  xargs -I{} -P "$JOBS" bash -lc '
    set -euo pipefail
    shard_path="$1"
    py="$2"
    tok_dir="$3"
    out_dir="$4"
    log_dir="$5"
    workers="$6"
    shard_name="$(basename "$shard_path" .jsonl)"
    out_path="$out_dir/${shard_name}.pt"
    log_path="$log_dir/${shard_name}.log"
    if [ -f "$out_path" ] && [ -f "${out_path}.meta.json" ]; then
      echo "skip_existing $shard_name" >> "$log_dir/_summary.log"
      exit 0
    fi
    echo "start $shard_name" >> "$log_dir/_summary.log"
    PYTHONPATH=src:scripts "$py" scripts/build_pretokenized_cache.py \
      --jsonl-source "$shard_path" \
      --tokenizer-dir "$tok_dir" \
      --output "$out_path" \
      --seq-len 512 \
      --workers "$workers" \
      > "$log_path" 2>&1
    echo "done $shard_name" >> "$log_dir/_summary.log"
  ' _ {} "$PY" "$TOK_DIR" "$OUT_DIR" "$LOG_DIR" "$WORKERS_PER_SHARD"

echo "stage=done" | tee -a "$ROOT/logs/sharded_setup.log"
