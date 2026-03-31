#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/wzhao20/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/wzhao20/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/wzhao20/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/wzhao20/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/wzhao20/torchinductor_cache}"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

PORT_BASE="${PORT_BASE:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_STEPS="${MAX_STEPS:-5}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
GPU_UTIL="${GPU_UTIL:-0.85}"
LOG_ROOT="${LOG_ROOT:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher}"
STALL_POLL_SECONDS="${STALL_POLL_SECONDS:-30}"
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-900}"
NEAR_COMPLETE_LINES="${NEAR_COMPLETE_LINES:-499}"
COMPLETE_EXIT_TIMEOUT_SECONDS="${COMPLETE_EXIT_TIMEOUT_SECONDS:-60}"
NEAR_COMPLETE_RETRIES="${NEAR_COMPLETE_RETRIES:-3}"
SEED_RETRIES="${SEED_RETRIES:-3}"
KILL_GRACE_SECONDS="${KILL_GRACE_SECONDS:-10}"

DEFAULT_DATASETS=(
  "/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/gsm_hard_500_20250507.json"
  "/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/math_500_20250414.json"
  "/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/aime_90_20250504.json"
  "/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/olymath_200_20250511.json"
)

DEFAULT_MODELS=(
  "Qwen/Qwen3-32B"
  "Qwen/Qwen3-14B"
)

DATASETS=("${DEFAULT_DATASETS[@]}")
MODELS=("${DEFAULT_MODELS[@]}")

if [[ -n "${DATASET_LIST:-}" ]]; then
  IFS=',' read -r -a DATASETS <<< "$DATASET_LIST"
fi

if [[ -n "${MODEL_LIST:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODEL_LIST"
fi

DEFAULT_SEEDS=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56)
SEEDS=("${DEFAULT_SEEDS[@]}")

if [[ -n "${SEED_LIST:-}" ]]; then
  IFS=',' read -r -a SEEDS <<< "$SEED_LIST"
fi

VLLM_PID=""

expected_line_count() {
  local data_path="$1"
  python - "$data_path" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
if isinstance(data, dict) and "examples" in data:
    print(len(data["examples"]))
else:
    print(len(data))
PY
}

cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "serve_vllm.py" 2>/dev/null || true
  sleep 5
}

wait_for_server() {
  local log_file="$1"
  local timeout_s="${2:-1800}"
  local waited=0
  until grep -q "Application startup complete." "$log_file" 2>/dev/null; do
    if (( waited >= timeout_s )); then
      echo "Timed out waiting for server startup: $log_file" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

result_jsonl_path() {
  local model_id="$1"
  local data_path="$2"
  local seed="$3"
  local model_name
  model_name="$(basename "$model_id")"
  local dataset_name
  dataset_name="$(basename "$data_path" .json)"
  printf "%s/%s_test/%s_temp=0.7_seed=%s_type=agent_steps=%s_python_only_python_only_seed%s.jsonl" \
    "$LOG_ROOT" "$dataset_name" "$model_name" "$seed" "$MAX_STEPS" "$seed"
}

is_completed_run() {
  local result_path="$1"
  local expected_lines="$2"
  if [[ -f "${result_path}.skip" ]]; then
    return 0
  fi
  [[ -f "$result_path" ]] || return 1
  local line_count
  line_count="$(wc -l < "$result_path" 2>/dev/null || echo 0)"
  [[ "$line_count" -ge "$expected_lines" ]]
}

current_line_count() {
  local result_path="$1"
  if [[ -f "$result_path" ]]; then
    wc -l < "$result_path" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

build_run_one_pass_cmd() {
  local model_id="$1"
  local data_path="$2"
  local seed="$3"

  local -a cmd=(
    python -m exps_research.unified_framework.run_experiment
    --experiment_type agent
    --data_path "$data_path"
    --model_type vllm
    --model_id "$model_id"
    --log_folder "$LOG_ROOT"
    --max_tokens "$MAX_TOKENS"
    --multithreading
    --use_process_pool
    --parallel_workers "$PARALLEL_WORKERS"
    --n 1
    --temperature 0.7
    --top_p 0.8
    --seed "$seed"
    --max_steps "$MAX_STEPS"
    --search_engine_type python_only
    --use_single_endpoint
    --suffix "python_only_seed${seed}"
  )
  printf '%q ' "${cmd[@]}"
}

terminate_run_group() {
  local pgid="$1"

  kill -TERM -- "-${pgid}" 2>/dev/null || true
  sleep "$KILL_GRACE_SECONDS"
  if kill -0 "$pgid" 2>/dev/null; then
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  fi
}

run_one_pass_with_retries() {
  local model_id="$1"
  local data_path="$2"
  local seed="$3"
  local result_path="$4"
  local expected_lines="$5"
  local default_near_complete_lines="$(( expected_lines > 1 ? expected_lines - 1 : expected_lines ))"
  local near_complete_lines="${NEAR_COMPLETE_LINES:-$default_near_complete_lines}"
  local attempt=0

  while (( attempt <= NEAR_COMPLETE_RETRIES )); do
    local before_lines last_lines stalled_for completed_for run_pid
    before_lines="$(current_line_count "$result_path")"
    last_lines="$before_lines"
    stalled_for=0
    completed_for=0

    local cmd_str
    cmd_str="$(build_run_one_pass_cmd "$model_id" "$data_path" "$seed")"
    setsid bash -lc "cd '$ROOT_DIR' && exec $cmd_str" &
    run_pid=$!

    while kill -0 "$run_pid" 2>/dev/null; do
      sleep "$STALL_POLL_SECONDS"
      local now_lines
      now_lines="$(current_line_count "$result_path")"

      if (( now_lines > last_lines )); then
        last_lines="$now_lines"
        stalled_for=0
      else
        stalled_for=$((stalled_for + STALL_POLL_SECONDS))
      fi

      if (( now_lines >= expected_lines )); then
        completed_for=$((completed_for + STALL_POLL_SECONDS))
      else
        completed_for=0
      fi

      if (( now_lines >= expected_lines && completed_for >= COMPLETE_EXIT_TIMEOUT_SECONDS )); then
        echo "Completed output for $model_id on $(basename "$data_path") seed=$seed at ${now_lines}/${expected_lines}, but process did not exit after ${completed_for}s. Terminating run group and continuing."
        terminate_run_group "$run_pid"
        wait "$run_pid" 2>/dev/null || true
        break
      fi

      if (( now_lines >= near_complete_lines && stalled_for >= STALL_TIMEOUT_SECONDS )); then
        echo "Stalled near completion for $model_id on $(basename "$data_path") seed=$seed at ${now_lines}/${expected_lines} after ${stalled_for}s."
        terminate_run_group "$run_pid"
        wait "$run_pid" 2>/dev/null || true
        break
      fi
    done

    if wait "$run_pid"; then
      if is_completed_run "$result_path" "$expected_lines"; then
        return 0
      fi
    fi

    local after_lines
    after_lines="$(current_line_count "$result_path")"
    if is_completed_run "$result_path" "$expected_lines"; then
      return 0
    fi
    if (( after_lines < near_complete_lines )); then
      return 1
    fi

    attempt=$((attempt + 1))
    if (( attempt > NEAR_COMPLETE_RETRIES )); then
      echo "Exceeded near-complete retries for $model_id on $(basename "$data_path") seed=$seed at ${after_lines}/${expected_lines}. Skipping this seed."
      touch "${result_path}.skip"
      return 0
    fi
    echo "Retrying near-complete $model_id on $(basename "$data_path") seed=$seed (attempt ${attempt}/${NEAR_COMPLETE_RETRIES})."
    pkill -f "run_experiment --experiment_type agent --data_path $data_path" 2>/dev/null || true
    sleep 5
  done
}

run_one_model() {
  local model_id="$1"
  local data_path="$2"
  local model_name
  model_name="$(basename "$model_id")"
  local dataset_name
  dataset_name="$(basename "$data_path" .json)"
  local serve_log="$LOG_ROOT/${model_name}_${dataset_name}_serve.log"
  local model_tp_size="$TP_SIZE"
  local model_gpu_util="$GPU_UTIL"
  local expected_lines
  expected_lines="$(expected_line_count "$data_path")"

  mkdir -p "$LOG_ROOT" "$LOG_ROOT/${dataset_name}_test"
  : > "$serve_log"

  cleanup

  case "$model_name" in
    Qwen3-32B)
      model_tp_size="${TP_SIZE_32B:-4}"
      model_gpu_util="${GPU_UTIL_32B:-0.85}"
      ;;
    Qwen3-14B)
      model_tp_size="${TP_SIZE_14B:-4}"
      model_gpu_util="${GPU_UTIL_14B:-0.50}"
      ;;
  esac

  python serve_vllm.py \
    --model "$model_id" \
    --tensor-parallel-size "$model_tp_size" \
    --port "$PORT_BASE" \
    --gpu-memory-utilization "$model_gpu_util" \
    --disable-log-requests \
    --disable-log-stats \
    > "$serve_log" 2>&1 &
  VLLM_PID=$!

  wait_for_server "$serve_log"

  for seed in "${SEEDS[@]}"; do
    local result_path
    result_path="$(result_jsonl_path "$model_id" "$data_path" "$seed")"
    if is_completed_run "$result_path" "$expected_lines"; then
      echo "=== Skipping completed $model_id on $(basename "$data_path") seed=$seed ==="
      continue
    fi
    echo "=== Generating $model_id on $(basename "$data_path") seed=$seed ==="
    local seed_attempt=1
    while (( seed_attempt <= SEED_RETRIES )); do
      if run_one_pass_with_retries "$model_id" "$data_path" "$seed" "$result_path" "$expected_lines"; then
        break
      fi
      echo "Seed-level retry for $model_id on $(basename "$data_path") seed=$seed (attempt ${seed_attempt}/${SEED_RETRIES})."
      seed_attempt=$((seed_attempt + 1))
      cleanup
      python serve_vllm.py \
        --model "$model_id" \
        --tensor-parallel-size "$model_tp_size" \
        --port "$PORT_BASE" \
        --gpu-memory-utilization "$model_gpu_util" \
        --disable-log-requests \
        --disable-log-stats \
        > "$serve_log" 2>&1 &
      VLLM_PID=$!
      wait_for_server "$serve_log"
    done

    if ! is_completed_run "$result_path" "$expected_lines"; then
      echo "Exceeded seed-level retries for $model_id on $(basename "$data_path") seed=$seed. Marking skip and continuing."
      touch "${result_path}.skip"
    fi
  done

  cleanup
  VLLM_PID=""
}

trap cleanup EXIT INT TERM

for data_path in "${DATASETS[@]}"; do
  for model_id in "${MODELS[@]}"; do
    run_one_model "$model_id" "$data_path"
  done
done
