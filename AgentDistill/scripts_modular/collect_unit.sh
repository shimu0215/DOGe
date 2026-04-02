#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

MODEL_ID=""
DATA_PATH=""
SEED=""
LORA_FOLDER=""
LOG_ROOT="/scratch/wzhao20/AKDA2/AgentDistill/logs/qa_results_python_only_teacher"
PORT_BASE="8000"
TP_SIZE="4"
MAX_TOKENS="1024"
MAX_STEPS="5"
PARALLEL_WORKERS="4"
GPU_UTIL="0.85"
MAX_LORA_RANK="64"
N="1"
FORCE_RERUN="0"
SERVER_TIMEOUT_SECONDS="1800"
API_BASE=""

usage() {
  cat <<'EOF'
Usage: collect_unit.sh --model-id MODEL --data-path DATA --seed SEED [options]

Required:
  --model-id            Base model id, e.g. Qwen/Qwen3-32B
  --data-path           Dataset json path
  --seed                Sampling seed

Optional:
  --lora-folder         LoRA path for fine-tuned teacher
  --log-root            Raw output root for base teachers
  --port-base           vLLM port
  --tp-size             Tensor parallel size
  --max-tokens          Generation max tokens
  --max-steps           Agent max steps
  --parallel-workers    run_experiment worker count
  --gpu-util            vLLM gpu-memory-utilization
  --max-lora-rank       vLLM max lora rank
  --n                   Number of samples per question
  --force-rerun         1 to ignore existing raw and recollect all
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --lora-folder) LORA_FOLDER="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --port-base) PORT_BASE="$2"; shift 2 ;;
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

if [[ -z "$MODEL_ID" || -z "$DATA_PATH" || -z "$SEED" ]]; then
  usage
  exit 1
fi

setup_agentdistill_env
cleanup_collection_resources

RESULT_JSONL="$(result_jsonl_path "$MODEL_ID" "$DATA_PATH" "$SEED" "$MAX_STEPS" "$N" "$LORA_FOLDER" "$LOG_ROOT")"
EXPECTED_COUNT="$(expected_question_count "$DATA_PATH")"
TASK_TYPE="$(infer_task_type "$DATA_PATH")"
SERVE_LOG_DIR="$(dirname "$RESULT_JSONL")"
mkdir -p "$SERVE_LOG_DIR"
SERVE_LOG="${SERVE_LOG_DIR}/$(basename "$MODEL_ID")_collect_seed${SEED}_serve.log"
API_BASE="http://127.0.0.1:${PORT_BASE}/v1"

RAW_BACKUP=""
REMAINING_DATA=""
REMAINING_RESULT_JSONL=""
VLLM_PID=""

resolve_remaining_result_path() {
  local preferred="$1"
  local model_id="$2"
  local remaining_data="$3"
  local seed="$4"
  local max_steps="$5"
  local n="$6"
  local log_root="$7"

  if [[ -f "$preferred" ]]; then
    echo "$preferred"
    return 0
  fi

  local model_name stem candidate
  model_name="$(basename "$model_id")"
  stem="$(basename "$remaining_data" .json)"
  candidate="$(find "$log_root" -maxdepth 3 -type f -name "${model_name}_temp=0.7*_seed=${seed}_type=agent_steps=${max_steps}_python_only_python_only_seed${seed}.jsonl" 2>/dev/null | grep "/${stem}_" | head -n 1 || true)"
  if [[ -n "$candidate" ]]; then
    echo "$candidate"
  else
    echo "$preferred"
  fi
}

restore_backup_on_failure() {
  if [[ -n "$RAW_BACKUP" && -f "$RAW_BACKUP" ]]; then
    if [[ ! -f "$RESULT_JSONL" ]]; then
      mv "$RAW_BACKUP" "$RESULT_JSONL"
    else
      local merged_tmp="${RESULT_JSONL}.restore.$$"
      merge_raw_results_by_question "$RAW_BACKUP" "$RESULT_JSONL" "$merged_tmp" >/dev/null
      mv "$merged_tmp" "$RESULT_JSONL"
      rm -f "$RAW_BACKUP"
    fi
  fi
}

cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  cleanup_collection_resources
  [[ -n "$REMAINING_DATA" && -f "$REMAINING_DATA" ]] && rm -f "$REMAINING_DATA" || true
}

trap 'restore_backup_on_failure; cleanup' EXIT INT TERM

if [[ "$FORCE_RERUN" == "1" ]]; then
  rm -f "$RESULT_JSONL"
fi

if is_collection_complete "$RESULT_JSONL" "$EXPECTED_COUNT"; then
  echo "Collection already complete: $RESULT_JSONL"
  exit 0
fi

REMAINING_DATA="$(mktemp /tmp/agentdistill_remaining.XXXXXX.json)"
if [[ -f "$RESULT_JSONL" ]]; then
  remaining="$(build_remaining_dataset "$DATA_PATH" "$RESULT_JSONL" "$REMAINING_DATA")"
  RAW_BACKUP="${RESULT_JSONL}.partial.$(date +%Y%m%d_%H%M%S)"
  mv "$RESULT_JSONL" "$RAW_BACKUP"
else
  cp "$DATA_PATH" "$REMAINING_DATA"
  remaining="$EXPECTED_COUNT"
fi

REMAINING_RESULT_JSONL="$(result_jsonl_path "$MODEL_ID" "$REMAINING_DATA" "$SEED" "$MAX_STEPS" "$N" "$LORA_FOLDER" "$LOG_ROOT")"

if (( remaining <= 0 )); then
  restore_backup_on_failure
  RAW_BACKUP=""
  exit 0
fi

: > "$SERVE_LOG"
VLLM_CMD=(
  "$PYTHON_BIN" serve_vllm.py
  --model "$MODEL_ID"
  --tensor-parallel-size "$TP_SIZE"
  --port "$PORT_BASE"
  --gpu-memory-utilization "$GPU_UTIL"
  --disable-log-requests
  --disable-log-stats
)

if [[ -n "$LORA_FOLDER" ]]; then
  VLLM_CMD+=(--lora-modules "finetune=$LORA_FOLDER" --max-lora-rank "$MAX_LORA_RANK")
fi

"${VLLM_CMD[@]}" > "$SERVE_LOG" 2>&1 &
VLLM_PID=$!
wait_for_server "$SERVE_LOG" "$SERVER_TIMEOUT_SECONDS"

RUN_CMD=(
  "$PYTHON_BIN" -m exps_research.unified_framework.run_experiment
  --experiment_type agent
  --task_type "$TASK_TYPE"
  --data_path "$REMAINING_DATA"
  --model_type vllm
  --model_id "$MODEL_ID"
  --api_base "$API_BASE"
  --log_folder "$LOG_ROOT"
  --max_tokens "$MAX_TOKENS"
  --multithreading
  --use_process_pool
  --parallel_workers "$PARALLEL_WORKERS"
  --n "$N"
  --temperature 0.7
  --top_p 0.8
  --seed "$SEED"
  --max_steps "$MAX_STEPS"
  --search_engine_type python_only
  --use_single_endpoint
  --suffix "python_only_seed${SEED}"
)

if [[ -n "$LORA_FOLDER" ]]; then
  RUN_CMD+=(--fine_tuned --lora_folder "$LORA_FOLDER")
fi

"${RUN_CMD[@]}"

if [[ -f "$REMAINING_RESULT_JSONL" ]]; then
  if [[ -f "$RESULT_JSONL" ]]; then
    MERGED_TMP="${RESULT_JSONL}.merged.$$"
    merge_raw_results_by_question "$RESULT_JSONL" "$REMAINING_RESULT_JSONL" "$MERGED_TMP" >/dev/null
    mv "$MERGED_TMP" "$RESULT_JSONL"
    rm -f "$REMAINING_RESULT_JSONL"
  else
    mkdir -p "$(dirname "$RESULT_JSONL")"
    mv "$REMAINING_RESULT_JSONL" "$RESULT_JSONL"
  fi
fi

REMAINING_RESULT_JSONL="$(resolve_remaining_result_path "$REMAINING_RESULT_JSONL" "$MODEL_ID" "$REMAINING_DATA" "$SEED" "$MAX_STEPS" "$N" "$LOG_ROOT")"

if [[ -f "$REMAINING_RESULT_JSONL" ]]; then
  if [[ -f "$RESULT_JSONL" ]]; then
    MERGED_TMP="${RESULT_JSONL}.merged.$$"
    merge_raw_results_by_question "$RESULT_JSONL" "$REMAINING_RESULT_JSONL" "$MERGED_TMP" >/dev/null
    mv "$MERGED_TMP" "$RESULT_JSONL"
    rm -f "$REMAINING_RESULT_JSONL"
  else
    mkdir -p "$(dirname "$RESULT_JSONL")"
    mv "$REMAINING_RESULT_JSONL" "$RESULT_JSONL"
  fi
fi
if [[ -n "$RAW_BACKUP" && -f "$RAW_BACKUP" ]]; then
  MERGED_TMP="${RESULT_JSONL}.merged.$$"
  merge_raw_results_by_question "$RAW_BACKUP" "$RESULT_JSONL" "$MERGED_TMP" >/dev/null
  mv "$MERGED_TMP" "$RESULT_JSONL"
  rm -f "$RAW_BACKUP"
  RAW_BACKUP=""
fi

if ! is_collection_complete "$RESULT_JSONL" "$EXPECTED_COUNT"; then
  echo "Collection still incomplete after run: $RESULT_JSONL" >&2
  exit 1
fi

echo "Collection complete: $RESULT_JSONL"
