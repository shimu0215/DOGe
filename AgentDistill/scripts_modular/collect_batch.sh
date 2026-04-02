#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

MODEL_ID=""
DATA_PATH=""
SEED_LIST=""
LORA_FOLDER=""
LOG_ROOT="/scratch/wzhao20/AKDA2/AgentDistill/logs/qa_results_python_only_teacher"
PORT_BASE_START="8000"
TP_SIZE="4"
MAX_TOKENS="1024"
MAX_STEPS="5"
PARALLEL_WORKERS="4"
GPU_UTIL="0.85"
MAX_LORA_RANK="64"
N="1"
FORCE_RERUN="0"

usage() {
  cat <<'EOF'
Usage: collect_batch.sh --model-id MODEL --data-path DATA --seed-list 42,43,44 [options]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --seed-list) SEED_LIST="$2"; shift 2 ;;
    --lora-folder) LORA_FOLDER="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --port-base-start) PORT_BASE_START="$2"; shift 2 ;;
    --tp-size) TP_SIZE="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --parallel-workers) PARALLEL_WORKERS="$2"; shift 2 ;;
    --gpu-util) GPU_UTIL="$2"; shift 2 ;;
    --max-lora-rank) MAX_LORA_RANK="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --force-rerun) FORCE_RERUN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL_ID" || -z "$DATA_PATH" || -z "$SEED_LIST" ]]; then
  usage
  exit 1
fi

setup_agentdistill_env
IFS=',' read -r -a SEEDS <<< "$SEED_LIST"

port="$PORT_BASE_START"
for seed in "${SEEDS[@]}"; do
  cleanup_collection_resources
  UNIT_CMD=(
    "${SCRIPT_DIR}/collect_unit.sh"
    --model-id "$MODEL_ID" \
    --data-path "$DATA_PATH" \
    --seed "$seed" \
    --log-root "$LOG_ROOT" \
    --port-base "$port" \
    --tp-size "$TP_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --max-steps "$MAX_STEPS" \
    --parallel-workers "$PARALLEL_WORKERS" \
    --gpu-util "$GPU_UTIL" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --n "$N" \
    --force-rerun "$FORCE_RERUN"
  )
  if [[ -n "$LORA_FOLDER" ]]; then
    UNIT_CMD+=(--lora-folder "$LORA_FOLDER")
  fi
  "${UNIT_CMD[@]}"
  cleanup_collection_resources
  port=$((port + 1))
done
