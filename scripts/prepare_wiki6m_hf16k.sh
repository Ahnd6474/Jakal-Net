#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/activate_env.sh"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

ROOT="${ROOT:-artifacts/wiki6m_hf16k}"
RAW_DIR="${RAW_DIR:-$ROOT/raw_shards}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$ROOT/tokenizer}"
PRETOK_DIR="${PRETOK_DIR:-$ROOT/pretokenized}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"

NUM_SHARDS="${NUM_SHARDS:-12}"
DOCS_PER_SHARD="${DOCS_PER_SHARD:-500000}"
DOWNLOAD_PARALLEL="${DOWNLOAD_PARALLEL:-6}"
TOKENIZER_SHARD_INDEX="${TOKENIZER_SHARD_INDEX:-0}"
VOCAB_SIZE="${VOCAB_SIZE:-16384}"
SEQ_LEN="${SEQ_LEN:-2048}"
PRETOK_PARALLEL="${PRETOK_PARALLEL:-2}"
PRETOK_WORKERS="${PRETOK_WORKERS:-8}"
PROCESSING_DEVICE="${PROCESSING_DEVICE:-}"

WIKI_SOURCE_SPEC="${WIKI_SOURCE_SPEC:-wiki=wikimedia/wikipedia|20231101.en|train|text|wiki}"

mkdir -p "$RAW_DIR" "$TOKENIZER_DIR" "$PRETOK_DIR" "$LOG_DIR"

echo "repo_root=$REPO_ROOT"
echo "root=$ROOT"
echo "num_shards=$NUM_SHARDS docs_per_shard=$DOCS_PER_SHARD download_parallel=$DOWNLOAD_PARALLEL"
echo "tokenizer_shard_index=$TOKENIZER_SHARD_INDEX vocab_size=$VOCAB_SIZE"
echo "seq_len=$SEQ_LEN pretoken_parallel=$PRETOK_PARALLEL pretoken_workers=$PRETOK_WORKERS"

wait_for_slot() {
    local limit="$1"
    while [ "$(jobs -rp | wc -l)" -ge "$limit" ]; do
        wait -n
    done
}

launch_download_shard() {
    local shard_index="$1"
    local shard_path
    local log_path
    shard_path="$(printf "%s/raw_shard_%04d.jsonl" "$RAW_DIR" "$shard_index")"
    log_path="$(printf "%s/download_shard_%04d.log" "$LOG_DIR" "$shard_index")"
    if [ -s "$shard_path" ]; then
        echo "skip_existing_download shard=$shard_index path=$shard_path"
        return 0
    fi
    python - "$shard_path" "$WIKI_SOURCE_SPEC" "$DOCS_PER_SHARD" "$NUM_SHARDS" "$shard_index" > "$log_path" 2>&1 <<'PY' &
import os
import runpy
import sys
import traceback
from pathlib import Path

output, source_spec, max_records, num_shards, shard_index = sys.argv[1:]
sys.argv = [
    "build_segmented_dialogue_corpus.py",
    "--output",
    output,
    "--no-default-mixed-dialogue-sources",
    "--no-default-document-sources",
    "--hf-document-source",
    source_spec,
    "--max-records-per-source",
    max_records,
    "--num-shards",
    num_shards,
    "--shard-index",
    shard_index,
]
try:
    runpy.run_path(str(Path("scripts/build_segmented_dialogue_corpus.py")), run_name="__main__")
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else 0
    os._exit(code)
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    os._exit(0)
PY
    echo "launched_download shard=$shard_index pid=$! log=$log_path"
}

for (( shard_index=0; shard_index<NUM_SHARDS; shard_index++ )); do
    wait_for_slot "$DOWNLOAD_PARALLEL"
    launch_download_shard "$shard_index"
done
wait

ROOT_ENV="$ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_ENV"])
raw_dir = root / "raw_shards"
total = 0
counts = {}
for path in sorted(raw_dir.glob("raw_shard_*.jsonl")):
    with path.open("r", encoding="utf-8") as handle:
        count = sum(1 for _ in handle)
    counts[path.name] = count
    total += count
summary = {"raw_dir": str(raw_dir), "total_documents": total, "per_shard_counts": counts}
out = root / "raw_shards_summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

TOKENIZER_INPUT="$(printf "%s/raw_shard_%04d.jsonl" "$RAW_DIR" "$TOKENIZER_SHARD_INDEX")"
TOKENIZER_JSON="$TOKENIZER_DIR/tokenizer.json"
if [ ! -f "$TOKENIZER_JSON" ]; then
    python scripts/train_hf_fast_bpe_tokenizer.py \
        --input "$TOKENIZER_INPUT" \
        --output-dir "$TOKENIZER_DIR" \
        --vocab-size "$VOCAB_SIZE" \
        > "$LOG_DIR/tokenizer_train.log" 2>&1
    echo "trained_tokenizer input=$TOKENIZER_INPUT output_dir=$TOKENIZER_DIR"
else
    echo "skip_existing_tokenizer path=$TOKENIZER_JSON"
fi

PRETOK_CMD=(
    python scripts/pretokenize_causal_memory_shards.py
    --mode orchestrate
    --raw-shard-dir "$RAW_DIR"
    --bundle-dir "$PRETOK_DIR"
    --num-shards "$NUM_SHARDS"
    --parallel-shards "$PRETOK_PARALLEL"
    --pretokenize-workers "$PRETOK_WORKERS"
    --tokenizer hf_auto
    --tokenizer-prefix "$ROOT/hf_auto_cache/wiki6m_hf16k"
    --hf-tokenizer-model "$TOKENIZER_DIR"
    --subword-vocab-size "$VOCAB_SIZE"
    --seq-len "$SEQ_LEN"
    --skip-existing
)

if [ -n "$PROCESSING_DEVICE" ]; then
    PRETOK_CMD+=(--processing-device "$PROCESSING_DEVICE")
fi

"${PRETOK_CMD[@]}" > "$LOG_DIR/pretokenize.log" 2>&1

ROOT_ENV="$ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_ENV"])
pretok_dir = root / "pretokenized"
paths = sorted(pretok_dir.glob("raw_shard_*.pt"))
summary = {
    "pretokenized_dir": str(pretok_dir),
    "shard_count": len(paths),
    "shards": [path.name for path in paths],
}
out = root / "pretokenized_summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
