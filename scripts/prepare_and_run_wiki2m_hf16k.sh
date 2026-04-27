#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Jakal-Net-ffn2-lion-24e5b7c
export PYTHONPATH=src:scripts
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-32}
export JAKAL_NET_FUSED_SCAN_DENSE_BLOCK_SIZE=${JAKAL_NET_FUSED_SCAN_DENSE_BLOCK_SIZE:-128}

PY=/workspace/Jakal-Net-docrun/.venv/bin/python
DATA=artifacts/data_wiki_2m/wiki_2m.jsonl
ROOT=artifacts/wiki2m_hf16k
TOK_DIR=$ROOT/tokenizer
PRETOK=$ROOT/pretokenized/wiki2m_hf_bpe16k_seq512.pt
BENCH=$ROOT/bench/random_batch_sizes.json
LOG_DIR=artifacts/training_logs
SEQ_SCRIPT=$ROOT/run_sequential_wiki2m_hf16k.sh
mkdir -p "$ROOT/logs" "$ROOT/pretokenized" "$LOG_DIR"

date -u | tee "$ROOT/logs/driver.start_utc.txt"

if [ ! -f "$TOK_DIR/tokenizer.json" ]; then
  echo "stage=tokenizer_train" | tee -a "$ROOT/logs/driver.log"
  "$PY" scripts/train_hf_fast_bpe_tokenizer.py \
    --input "$DATA" \
    --output-dir "$TOK_DIR" \
    --vocab-size 16384 \
    > "$ROOT/logs/tokenizer_train.log" 2>&1
else
  echo "stage=tokenizer_train skip_existing" | tee -a "$ROOT/logs/driver.log"
fi

VOCAB_SIZE=$("$PY" - <<PY
from transformers import AutoTokenizer
print(len(AutoTokenizer.from_pretrained("$TOK_DIR", use_fast=True)))
PY
)
echo "vocab_size=$VOCAB_SIZE" | tee -a "$ROOT/logs/driver.log"

if [ ! -f "$PRETOK" ]; then
  echo "stage=pretokenize" | tee -a "$ROOT/logs/driver.log"
  "$PY" scripts/build_pretokenized_cache.py \
    --jsonl-source "$DATA" \
    --tokenizer-dir "$TOK_DIR" \
    --output "$PRETOK" \
    --seq-len 512 \
    --workers 16 \
    > "$ROOT/logs/pretokenize.log" 2>&1
else
  echo "stage=pretokenize skip_existing" | tee -a "$ROOT/logs/driver.log"
fi

if [ ! -f "$BENCH" ]; then
  echo "stage=batch_benchmark" | tee -a "$ROOT/logs/driver.log"
  "$PY" scripts/benchmark_random_batch_sizes.py \
    --vocab-size "$VOCAB_SIZE" \
    --seq-len 512 \
    --dim 384 \
    --precision bf16 \
    --max-batch 128 \
    --output "$BENCH" \
    > "$ROOT/logs/batch_benchmark.log" 2>&1
else
  echo "stage=batch_benchmark skip_existing" | tee -a "$ROOT/logs/driver.log"
fi

read -r B_TRANS B_NOMEM B_NOREAD < <("$PY" - <<PY
import json
p="$BENCH"
d=json.load(open(p))
print(d["transformer"]["recommended_batch"], d["nomemory"]["recommended_batch"], d["noread"]["recommended_batch"])
PY
)
echo "recommended_batches transformer=$B_TRANS nomemory=$B_NOMEM noread=$B_NOREAD" | tee -a "$ROOT/logs/driver.log"

cat > "$SEQ_SCRIPT" <<SEQ
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Jakal-Net-ffn2-lion-24e5b7c
export PYTHONPATH=src:scripts
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=32
export JAKAL_NET_FUSED_SCAN_DENSE_BLOCK_SIZE=128
PY=/workspace/Jakal-Net-docrun/.venv/bin/python
PRETOK=$PRETOK
LOG_DIR=artifacts/training_logs
mkdir -p "\$LOG_DIR"

run_transformer() {
  "\$PY" scripts/train_causal_memory_lm.py \
    --pretokenized-path "\$PRETOK" \
    --device cuda --precision bf16 \
    --model-kind transformer --transformer-layers 5 --transformer-heads 6 --transformer-ff-mult 3.7005 --transformer-dropout 0.0 \
    --batch-size $B_TRANS --grad-accum-steps 1 --stage1-batch-size $B_TRANS --stage2-batch-size $B_TRANS --stage3-batch-size $B_TRANS \
    --learning-rate 0.0002 --warmup-start-lr 0.00003 --warmup-steps 40 --lr-decay-start-step 50 --lr-decay-steps 180 --lr-min-ratio 0.25 \
    --optimizer adamw_fused --epochs 1.0 --grad-clip 1.0 --diagnose-nonfinite-grad --diagnose-nonfinite-limit 6 \
    --eval-start-step 500 --eval-interval 500 --checkpoint-interval 200 --eval-sample-interval 1000 --eval-documents 8 \
    --curriculum-stage1-ratio 0.0075 --curriculum-stage2-ratio 0.0125 --curriculum-stage1-span 1 --curriculum-stage2-span 1 --curriculum-stage3-span 1 \
    --run-name wiki2m_hf16k_transformer_b${B_TRANS}_adamw_eval500_v1 --tensorboard \
    > "\$LOG_DIR/wiki2m_hf16k_transformer_b${B_TRANS}_adamw_eval500_v1.log" 2>&1
}

run_nomemory() {
  "\$PY" scripts/train_causal_memory_lm.py \
    --pretokenized-path "\$PRETOK" \
    --device cuda --precision bf16 \
    --model-kind causal_memory --s-layers 6 --memory-slots 384 96 24 --memory-update-intervals 1 2 4 --prediction-layers 3 \
    --s-window 256 --prediction-window 64 --feed-forward-hidden-mult 2 --memory-topk 16 --memory-train-mode dense --memory-eval-mode topk --eval-topk 16 \
    --scan-backend auto --enable-fused-training --enable-scan-backward-cuda \
    --pairwise-kind low_rank_bilinear --route-kind low_rank_bilinear --pairwise-rank 128 --route-rank 96 --pairwise-heads 4 --route-heads 4 --implementation native \
    --batch-size $B_NOMEM --grad-accum-steps 1 --stage1-batch-size $B_NOMEM --stage2-batch-size $B_NOMEM --stage3-batch-size $B_NOMEM \
    --learning-rate 0.0002 --warmup-start-lr 0.00003 --warmup-steps 40 --lr-decay-start-step 50 --lr-decay-steps 180 --lr-min-ratio 0.25 \
    --optimizer lion --epochs 1.0 --grad-clip 1.0 --diagnose-nonfinite-grad --diagnose-nonfinite-limit 6 \
    --eval-start-step 500 --eval-interval 500 --checkpoint-interval 200 --eval-sample-interval 1000 --eval-documents 8 \
    --curriculum-stage1-ratio 0.0075 --curriculum-stage2-ratio 0.0125 --curriculum-stage1-span 1 --curriculum-stage2-span 1 --curriculum-stage3-span 1 \
    --disable-feed-forward-layers --disable-memory \
    --run-name wiki2m_hf16k_nomemory_b${B_NOMEM}_lion_eval500_v1 --tensorboard \
    > "\$LOG_DIR/wiki2m_hf16k_nomemory_b${B_NOMEM}_lion_eval500_v1.log" 2>&1
}

run_noread() {
  "\$PY" scripts/train_causal_memory_lm.py \
    --pretokenized-path "\$PRETOK" \
    --device cuda --precision bf16 \
    --model-kind causal_memory --s-layers 6 --memory-slots 384 96 24 --memory-update-intervals 1 2 4 --prediction-layers 3 \
    --s-window 256 --prediction-window 64 --feed-forward-hidden-mult 2 --memory-topk 16 --memory-train-mode dense --memory-eval-mode topk --eval-topk 16 \
    --scan-backend auto --enable-fused-training --enable-scan-backward-cuda \
    --pairwise-kind low_rank_bilinear --route-kind low_rank_bilinear --pairwise-rank 128 --route-rank 96 --pairwise-heads 4 --route-heads 4 --implementation native \
    --batch-size $B_NOREAD --grad-accum-steps 1 --stage1-batch-size $B_NOREAD --stage2-batch-size $B_NOREAD --stage3-batch-size $B_NOREAD \
    --learning-rate 0.0002 --warmup-start-lr 0.00003 --warmup-steps 40 --lr-decay-start-step 50 --lr-decay-steps 180 --lr-min-ratio 0.25 \
    --optimizer lion --epochs 1.0 --grad-clip 1.0 --diagnose-nonfinite-grad --diagnose-nonfinite-limit 6 \
    --eval-start-step 500 --eval-interval 500 --checkpoint-interval 200 --eval-sample-interval 1000 --eval-documents 8 \
    --curriculum-stage1-ratio 0.0075 --curriculum-stage2-ratio 0.0125 --curriculum-stage1-span 1 --curriculum-stage2-span 1 --curriculum-stage3-span 1 \
    --disable-feed-forward-layers --disable-memory-read \
    --run-name wiki2m_hf16k_noread_b${B_NOREAD}_lion_eval500_v1 --tensorboard \
    > "\$LOG_DIR/wiki2m_hf16k_noread_b${B_NOREAD}_lion_eval500_v1.log" 2>&1
}

run_transformer
run_nomemory
run_noread
SEQ
chmod +x "$SEQ_SCRIPT"

echo "stage=sequential_training" | tee -a "$ROOT/logs/driver.log"
bash "$SEQ_SCRIPT" > "$ROOT/logs/sequential_driver.log" 2>&1

date -u | tee "$ROOT/logs/driver.done_utc.txt"
