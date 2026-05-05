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
MAX_TOKENS="256"
MAX_STEPS="5"
PARALLEL_WORKERS="4"
GPU_UTIL="0.85"
MAX_LORA_RANK="64"
N="1"
TEMPERATURE="0.7"
SAVE_LOGPROBS="0"
TOP_LOGPROBS=""
FORCE_RERUN="0"
SERVER_TIMEOUT_SECONDS="1800"
API_BASE=""
PER_TASK_TIMEOUT="0"
OUTPUT_NAME_TAG=""
ANSWER_TOOL_PROMPT_NAME="final_answer"
VLLM_START_RETRIES="3"
GPU_UTIL_FALLBACK_STEP="0.05"
AUTO_TP_UPSCALE="1"
REQUEST_TIMEOUT="600"

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
  --request-timeout     Model request timeout in seconds (vLLM/OpenAI client timeout)
  --parallel-workers    run_experiment worker count
  --gpu-util            vLLM gpu-memory-utilization
  --max-lora-rank       vLLM max lora rank
  --n                   Number of samples per question
  --temperature         Sampling temperature
  --save-logprobs       1 to request token logprobs and save them with trajectories
  --top-logprobs        Number of top logprobs to save per generated token
  --force-rerun         1 to ignore existing raw and recollect all
  --per-task-timeout    Per-question timeout in seconds for outer process-pool guard; <=0 disables it
  --output-name-tag     Stable name tag used for result folder/file naming
  --answer-tool-prompt-name
                        Prompt-only alias for final answer tool name (default: final_answer)
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
    --request-timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
    --parallel-workers) PARALLEL_WORKERS="$2"; shift 2 ;;
    --gpu-util) GPU_UTIL="$2"; shift 2 ;;
    --max-lora-rank) MAX_LORA_RANK="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --save-logprobs) SAVE_LOGPROBS="$2"; shift 2 ;;
    --top-logprobs) TOP_LOGPROBS="$2"; shift 2 ;;
    --force-rerun) FORCE_RERUN="$2"; shift 2 ;;
    --per-task-timeout) PER_TASK_TIMEOUT="$2"; shift 2 ;;
    --output-name-tag) OUTPUT_NAME_TAG="$2"; shift 2 ;;
    --answer-tool-prompt-name) ANSWER_TOOL_PROMPT_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL_ID" || -z "$DATA_PATH" || -z "$SEED" ]]; then
  usage
  exit 1
fi

if (( PER_TASK_TIMEOUT > 0 )); then
  timeout_floor=300
  if [[ "$MAX_STEPS" =~ ^[0-9]+$ ]] && (( MAX_STEPS > 0 )); then
    adaptive_floor=$(( MAX_STEPS * 90 ))
    if (( adaptive_floor > timeout_floor )); then
      timeout_floor=$adaptive_floor
    fi
  fi
  if (( PER_TASK_TIMEOUT < timeout_floor )); then
    echo "per_task_timeout=$PER_TASK_TIMEOUT is too aggressive for multi-step agent eval; raising to ${timeout_floor}s."
    PER_TASK_TIMEOUT=$timeout_floor
  fi
fi

# Keep request timeout aligned with outer per-task timeout when provided.
if (( PER_TASK_TIMEOUT > 0 && REQUEST_TIMEOUT < PER_TASK_TIMEOUT )); then
  REQUEST_TIMEOUT="$PER_TASK_TIMEOUT"
fi

setup_agentdistill_env
cleanup_collection_resources

if [[ -z "$OUTPUT_NAME_TAG" ]]; then
  OUTPUT_NAME_TAG="$(basename "$DATA_PATH" .json)"
fi

SUFFIX_TAG="python_only_seed${SEED}"
if [[ "$SAVE_LOGPROBS" == "1" ]]; then
  if [[ -z "$TOP_LOGPROBS" ]]; then
    echo "--top-logprobs is required when --save-logprobs=1" >&2
    exit 1
  fi
  SUFFIX_TAG="python_only_toplogprobs${TOP_LOGPROBS}_seed${SEED}"
fi

RESULT_JSONL="$(result_jsonl_path "$MODEL_ID" "$DATA_PATH" "$SEED" "$MAX_STEPS" "$N" "$LORA_FOLDER" "$LOG_ROOT" "$OUTPUT_NAME_TAG" "$TEMPERATURE" "$SUFFIX_TAG")"
EXPECTED_COUNT="$(expected_question_count "$DATA_PATH")"
TASK_TYPE="$(infer_task_type "$DATA_PATH")"
SERVE_LOG_DIR="$(dirname "$RESULT_JSONL")"
DATASET_NAME="$OUTPUT_NAME_TAG"
mkdir -p "$SERVE_LOG_DIR"
SERVE_LOG="${SERVE_LOG_DIR}/$(basename "$MODEL_ID")_${DATASET_NAME}_collect_seed${SEED}_serve.log"
PORT_BASE="8000"
API_BASE="http://127.0.0.1:8000/v1"

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
  stem="$OUTPUT_NAME_TAG"
  local suffix_tag="$8"
  candidate="$(find "$log_root" -maxdepth 3 -type f -name "${model_name}_${stem}_temp=${TEMPERATURE}*_seed=${seed}_type=agent_steps=${max_steps}_${suffix_tag}.jsonl" 2>/dev/null | head -n 1 || true)"
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

REMAINING_RESULT_JSONL="$(result_jsonl_path "$MODEL_ID" "$REMAINING_DATA" "$SEED" "$MAX_STEPS" "$N" "$LORA_FOLDER" "$LOG_ROOT" "$OUTPUT_NAME_TAG" "$TEMPERATURE" "$SUFFIX_TAG")"

if (( remaining <= 0 )); then
  restore_backup_on_failure
  RAW_BACKUP=""
  exit 0
fi

visible_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local IFS=','
    read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    echo "${#gpu_ids[@]}"
    return 0
  fi
  nvidia-smi -L 2>/dev/null | wc -l
}

start_vllm_with_retries() {
  local retries="$1"
  local attempt=1
  local available_gpus
  available_gpus="$(visible_gpu_count)"

  while (( attempt <= retries )); do
    local util tp_candidate
    util="$(awk -v base="$GPU_UTIL" -v step="$GPU_UTIL_FALLBACK_STEP" -v idx="$((attempt - 1))" 'BEGIN{u=base-step*idx; if (u<0.55) u=0.55; printf "%.2f", u}')"
    tp_candidate="$TP_SIZE"

    if [[ "$AUTO_TP_UPSCALE" == "1" && "$TP_SIZE" -eq 1 && "$available_gpus" -ge 2 && "$attempt" -ge 2 ]]; then
      tp_candidate=2
    fi
    if (( tp_candidate > available_gpus )); then
      tp_candidate="$available_gpus"
    fi
    if (( tp_candidate < 1 )); then
      tp_candidate=1
    fi

    : > "$SERVE_LOG"
    VLLM_CMD=(
      "$PYTHON_BIN" serve_vllm.py
      --model "$MODEL_ID"
      --tensor-parallel-size "$tp_candidate"
      --port "$PORT_BASE"
      --gpu-memory-utilization "$util"
      --disable-log-requests
      --disable-log-stats
    )
    if [[ -n "$LORA_FOLDER" ]]; then
      VLLM_CMD+=(--lora-modules "finetune=$LORA_FOLDER" --max-lora-rank "$MAX_LORA_RANK")
    fi

    echo "Starting vLLM (attempt $attempt/$retries): tp=$tp_candidate gpu_util=$util"
    "${VLLM_CMD[@]}" > "$SERVE_LOG" 2>&1 &
    VLLM_PID=$!

    if wait_for_server "$SERVE_LOG" "$SERVER_TIMEOUT_SECONDS" "$VLLM_PID"; then
      echo "vLLM startup succeeded."
      return 0
    fi

    echo "vLLM startup failed on attempt $attempt. Cleaning up and retrying..."
    cleanup_collection_resources
    VLLM_PID=""
    attempt=$((attempt + 1))
  done

  return 1
}

if ! start_vllm_with_retries "$VLLM_START_RETRIES"; then
  echo "Failed to start vLLM after $VLLM_START_RETRIES attempts. See log: $SERVE_LOG" >&2
  exit 1
fi

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
  --request_timeout "$REQUEST_TIMEOUT"
  --multithreading
  --use_process_pool
  --parallel_workers "$PARALLEL_WORKERS"
  --per_task_timeout "$PER_TASK_TIMEOUT"
  --output_name_tag "$OUTPUT_NAME_TAG"
  --n "$N"
  --temperature "$TEMPERATURE"
  --top_p 0.8
  --seed "$SEED"
  --max_steps "$MAX_STEPS"
  --search_engine_type python_only
  --use_single_endpoint
  --suffix "$SUFFIX_TAG"
)

if "$PYTHON_BIN" -m exps_research.unified_framework.run_experiment -h 2>&1 | grep -q -- "--answer_tool_prompt_name"; then
  RUN_CMD+=(--answer_tool_prompt_name "$ANSWER_TOOL_PROMPT_NAME")
fi

if [[ -n "$LORA_FOLDER" ]]; then
  RUN_CMD+=(--fine_tuned --lora_folder "$LORA_FOLDER")
fi

if [[ "$SAVE_LOGPROBS" == "1" ]]; then
  RUN_CMD+=(--save_logprobs --top_logprobs "$TOP_LOGPROBS")
fi

"${RUN_CMD[@]}"

if [[ -f "$REMAINING_RESULT_JSONL" ]]; then
  if [[ "$REMAINING_RESULT_JSONL" == "$RESULT_JSONL" ]]; then
    : # same file — run_experiment wrote directly to RESULT_JSONL; nothing to merge
  elif [[ -f "$RESULT_JSONL" ]]; then
    MERGED_TMP="${RESULT_JSONL}.merged.$$"
    merge_raw_results_by_question "$RESULT_JSONL" "$REMAINING_RESULT_JSONL" "$MERGED_TMP" >/dev/null
    mv "$MERGED_TMP" "$RESULT_JSONL"
    rm -f "$REMAINING_RESULT_JSONL"
  else
    mkdir -p "$(dirname "$RESULT_JSONL")"
    mv "$REMAINING_RESULT_JSONL" "$RESULT_JSONL"
  fi
fi

REMAINING_RESULT_JSONL="$(resolve_remaining_result_path "$REMAINING_RESULT_JSONL" "$MODEL_ID" "$REMAINING_DATA" "$SEED" "$MAX_STEPS" "$N" "$LOG_ROOT" "$SUFFIX_TAG")"

if [[ -f "$REMAINING_RESULT_JSONL" ]]; then
  if [[ "$REMAINING_RESULT_JSONL" == "$RESULT_JSONL" ]]; then
    : # same file — already in place
  elif [[ -f "$RESULT_JSONL" ]]; then
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
