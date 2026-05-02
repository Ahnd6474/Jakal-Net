#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Jakal-Net
ROOT=artifacts/wiki2m_hf16k
TOK_FILE=$ROOT/tokenizer/tokenizer.json
WATCH_LOG=$ROOT/logs/sharded_watcher.log
SHARD_LOG=$ROOT/logs/sharded_driver.out

mkdir -p "$ROOT/logs"

echo "watch_start" >> "$WATCH_LOG"
while [ ! -f "$TOK_FILE" ]; do
  sleep 10
done
echo "tokenizer_ready" >> "$WATCH_LOG"

pkill -f "build_pretokenized_cache.py.*wiki2m_hf_bpe16k_seq512.pt" || true
pkill -f "bash scripts/setup_wiki2m_hf16k_remote.sh" || true

if pgrep -f "bash scripts/build_pretokenized_shards_remote.sh" >/dev/null; then
  echo "sharded_builder_already_running" >> "$WATCH_LOG"
  exit 0
fi

nohup bash scripts/build_pretokenized_shards_remote.sh > "$SHARD_LOG" 2>&1 < /dev/null &
echo "sharded_builder_started pid=$!" >> "$WATCH_LOG"
