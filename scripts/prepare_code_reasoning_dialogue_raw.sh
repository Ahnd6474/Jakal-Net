#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"

export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/data_mix_raw_v2}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"

mkdir -p "${OUTPUT_DIR}/logs"

python scripts/build_code_reasoning_dialogue_raw_parallel.py \
  --output-dir "${OUTPUT_DIR}" \
  --max-parallel "${MAX_PARALLEL}"
