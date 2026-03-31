#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TARGET_ROOT="${TARGET_ROOT:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test}"
DATASET_PATH="${DATASET_PATH:-/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/math_500_20250414.json}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
SEED_LIST="${SEED_LIST:-45,46,47,48,49,50,51,52,53,54,55,56}"
BACKUP_ROOT="${TARGET_ROOT}/rerun_backup_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BACKUP_ROOT/raw" "$BACKUP_ROOT/evaluations" "$BACKUP_ROOT/filtered_data"

IFS=',' read -r -a SEEDS <<< "$SEED_LIST"
for s in "${SEEDS[@]}"; do
  raw="${TARGET_ROOT}/Qwen3-32B_temp=0.7_seed=${s}_type=agent_steps=5_python_only_python_only_seed${s}.jsonl"
  scored="${TARGET_ROOT}/evaluations/Qwen3-32B_temp=0.7_seed=${s}_type=agent_steps=5_python_only_python_only_seed${s}_scored.jsonl"
  filtered="${TARGET_ROOT}/filtered_data/Qwen3-32B_temp=0.7_seed=${s}_type=agent_steps=5_python_only_python_only_seed${s}_filtered.jsonl"

  if [[ -f "$raw" ]]; then
    mv "$raw" "$BACKUP_ROOT/raw/"
  fi
  if [[ -f "${raw}.skip" ]]; then
    mv "${raw}.skip" "$BACKUP_ROOT/raw/"
  fi
  if [[ -f "$scored" ]]; then
    mv "$scored" "$BACKUP_ROOT/evaluations/"
  fi
  if [[ -f "$filtered" ]]; then
    mv "$filtered" "$BACKUP_ROOT/filtered_data/"
  fi
done

echo "Backed up failed files to: $BACKUP_ROOT"

export DATASET_LIST="$DATASET_PATH"
export MODEL_LIST="$MODEL_ID"
export SEED_LIST
export STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-180}"
export STALL_POLL_SECONDS="${STALL_POLL_SECONDS:-15}"
export NEAR_COMPLETE_LINES="${NEAR_COMPLETE_LINES:-499}"
export COMPLETE_EXIT_TIMEOUT_SECONDS="${COMPLETE_EXIT_TIMEOUT_SECONDS:-30}"
export NEAR_COMPLETE_RETRIES="${NEAR_COMPLETE_RETRIES:-5}"
export SEED_RETRIES="${SEED_RETRIES:-5}"

bash scripts/inference/run_agent_python_only_data_sequence.sh
