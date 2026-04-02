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

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
LORA_FOLDER="${LORA_FOLDER:-}"
DATA_PATH="${DATA_PATH:-/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/math_500_20250414.json}"
DEFAULT_LOG_ROOT="/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher"
if [[ -n "${LORA_FOLDER:-}" ]]; then
  LOG_ROOT="${LOG_ROOT:-$LORA_FOLDER/qa_results}"
else
  LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
fi
SEED="${SEED:-42}"
PORT_BASE="${PORT_BASE:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_STEPS="${MAX_STEPS:-5}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
N="${N:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
STALL_POLL_SECONDS="${STALL_POLL_SECONDS:-15}"
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-180}"
NEAR_COMPLETE_LINES="${NEAR_COMPLETE_LINES:-499}"
COMPLETE_EXIT_TIMEOUT_SECONDS="${COMPLETE_EXIT_TIMEOUT_SECONDS:-30}"
NEAR_COMPLETE_RETRIES="${NEAR_COMPLETE_RETRIES:-5}"
KILL_GRACE_SECONDS="${KILL_GRACE_SECONDS:-10}"

VLLM_PID=""
RUN_DATA_PATH=""
TMP_DATA_PATH=""

cleanup() {
  if [[ -n "${TMP_DATA_PATH}" && -f "${TMP_DATA_PATH}" ]]; then
    rm -f "${TMP_DATA_PATH}" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "serve_vllm.py" 2>/dev/null || true
  sleep 3
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

current_line_count() {
  if [[ -f "$RESULT_JSONL" ]]; then
    wc -l < "$RESULT_JSONL" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

current_unique_question_count() {
  python - "$RESULT_JSONL" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit

seen = set()
with path.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        question = entry.get("question") or entry.get("problem") or entry.get("prompt")
        if question:
            seen.add(question)
print(len(seen))
PY
}

terminate_run_group() {
  local pgid="$1"
  kill -TERM -- "-${pgid}" 2>/dev/null || true
  sleep "$KILL_GRACE_SECONDS"
  if kill -0 "$pgid" 2>/dev/null; then
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  fi
}

expected_line_count() {
  python - "$DATA_PATH" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
if isinstance(data, dict) and "examples" in data:
    print(len(data["examples"]))
else:
    print(len(data))
PY
}

build_remaining_dataset() {
  local source_data_path="$1"
  local result_jsonl="$2"
  local output_json="$3"
  python - "$source_data_path" "$result_jsonl" "$output_json" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

with source_path.open() as f:
    source = json.load(f)

if isinstance(source, dict) and "examples" in source:
    examples = source["examples"]
    wrap = True
else:
    examples = source
    wrap = False

completed = set()
if result_path.exists():
    with result_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            question = entry.get("question") or entry.get("problem") or entry.get("prompt")
            if question:
                completed.add(question)

remaining = []
for entry in examples:
    question = entry.get("question") or entry.get("problem") or entry.get("prompt")
    if question in completed:
        continue
    remaining.append(entry)

payload = dict(source) if wrap else remaining
if wrap:
    payload["examples"] = remaining

with output_path.open("w") as f:
    json.dump(payload, f)

print(len(remaining))
PY
}

result_jsonl_path() {
  local model_name dataset_name base_dir
  model_name="$(basename "$MODEL_ID")"
  dataset_name="$(basename "$DATA_PATH" .json)"
  if [[ -n "$LORA_FOLDER" ]]; then
    base_dir="$LORA_FOLDER/qa_results"
    printf "%s/%s_test/%s_temp=0.7_n=%s_seed=%s_type=agent_steps=%s_python_only_python_only_seed%s.jsonl" \
      "$base_dir" "$dataset_name" "$model_name" "$N" "$SEED" "$MAX_STEPS" "$SEED"
  else
    base_dir="$LOG_ROOT"
    printf "%s/%s_test/%s_temp=0.7_seed=%s_type=agent_steps=%s_python_only_python_only_seed%s.jsonl" \
      "$base_dir" "$dataset_name" "$model_name" "$SEED" "$MAX_STEPS" "$SEED"
  fi
}

trap cleanup EXIT INT TERM

RESULT_JSONL="$(result_jsonl_path)"
EXPECTED_LINES="$(expected_line_count)"
SERVE_LOG_DIR="$(dirname "$RESULT_JSONL")"
mkdir -p "$SERVE_LOG_DIR"
SERVE_LOG="$SERVE_LOG_DIR/$(basename "$MODEL_ID")_teacher_collect_seed${SEED}_serve.log"

if [[ -f "$RESULT_JSONL" ]]; then
  existing_lines="$(current_line_count)"
  existing_unique_questions="$(current_unique_question_count)"
  if [[ "$FORCE_RERUN" == "1" ]]; then
    rm -f "$RESULT_JSONL"
  elif [[ "$existing_unique_questions" -ge "$EXPECTED_LINES" ]]; then
    echo "Teacher collection already completed at $RESULT_JSONL ($existing_lines rows, $existing_unique_questions unique questions)."
    exit 0
  fi
fi

cleanup
: > "$SERVE_LOG"

CMD=(
  python serve_vllm.py
  --model "$MODEL_ID"
  --tensor-parallel-size "$TP_SIZE"
  --port "$PORT_BASE"
  --gpu-memory-utilization "$GPU_UTIL"
  --disable-log-requests
  --disable-log-stats
)

if [[ -n "$LORA_FOLDER" ]]; then
  CMD+=(--lora-modules "finetune=$LORA_FOLDER" --max-lora-rank "$MAX_LORA_RANK")
fi

"${CMD[@]}" > "$SERVE_LOG" 2>&1 &
VLLM_PID=$!

wait_for_server "$SERVE_LOG"

run_one_pass_with_retries() {
  local expected_lines="$1"
  local default_near_complete_lines="$(( expected_lines > 1 ? expected_lines - 1 : expected_lines ))"
  local near_complete_lines="${NEAR_COMPLETE_LINES:-$default_near_complete_lines}"
  local attempt=0

  while (( attempt <= NEAR_COMPLETE_RETRIES )); do
    local last_unique_questions stalled_for completed_for run_pid
    last_unique_questions="$(current_unique_question_count)"
    stalled_for=0
    completed_for=0

    if [[ -n "${TMP_DATA_PATH}" && -f "${TMP_DATA_PATH}" ]]; then
      rm -f "${TMP_DATA_PATH}" 2>/dev/null || true
    fi
    TMP_DATA_PATH="$(mktemp /tmp/teacher_collect_remaining.XXXXXX.json)"
    remaining_questions="$(build_remaining_dataset "$DATA_PATH" "$RESULT_JSONL" "$TMP_DATA_PATH")"
    RUN_DATA_PATH="$TMP_DATA_PATH"
    echo "Remaining questions to collect for $(basename "$MODEL_ID") seed=$SEED: ${remaining_questions}"
    if (( remaining_questions <= 0 )); then
      return 0
    fi

    RUN_CMD=(
      python -m exps_research.unified_framework.run_experiment
      --experiment_type agent
      --data_path "$RUN_DATA_PATH"
      --model_type vllm
      --model_id "$MODEL_ID"
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
      --task_type math
      --suffix "python_only_seed${SEED}"
    )

    if [[ -n "$LORA_FOLDER" ]]; then
      RUN_CMD+=(--fine_tuned --lora_folder "$LORA_FOLDER")
    fi

    printf 'RUN_CMD:'
    printf ' %q' "${RUN_CMD[@]}"
    printf '\n'

    setsid "${RUN_CMD[@]}" &
    run_pid=$!

    while kill -0 "$run_pid" 2>/dev/null; do
      sleep "$STALL_POLL_SECONDS"
      local now_unique_questions
      now_unique_questions="$(current_unique_question_count)"

      if (( now_unique_questions > last_unique_questions )); then
        last_unique_questions="$now_unique_questions"
        stalled_for=0
      else
        stalled_for=$((stalled_for + STALL_POLL_SECONDS))
      fi

      if (( now_unique_questions >= expected_lines )); then
        completed_for=$((completed_for + STALL_POLL_SECONDS))
      else
        completed_for=0
      fi

      if (( now_unique_questions >= expected_lines && completed_for >= COMPLETE_EXIT_TIMEOUT_SECONDS )); then
        echo "Completed output for $(basename "$MODEL_ID") seed=$SEED at ${now_unique_questions}/${expected_lines} unique questions, but process did not exit after ${completed_for}s. Terminating run group and continuing."
        terminate_run_group "$run_pid"
        wait "$run_pid" 2>/dev/null || true
        return 0
      fi

      if (( now_unique_questions >= near_complete_lines && stalled_for >= STALL_TIMEOUT_SECONDS )); then
        echo "Stalled near completion for $(basename "$MODEL_ID") seed=$SEED at ${now_unique_questions}/${expected_lines} unique questions after ${stalled_for}s."
        terminate_run_group "$run_pid"
        wait "$run_pid" 2>/dev/null || true
        break
      fi
    done

    if wait "$run_pid"; then
      local final_unique_questions
      final_unique_questions="$(current_unique_question_count)"
      if (( final_unique_questions >= expected_lines )); then
        return 0
      fi
    fi

    local after_unique_questions
    after_unique_questions="$(current_unique_question_count)"
    if (( after_unique_questions >= expected_lines )); then
      return 0
    fi
    if (( after_unique_questions < near_complete_lines )); then
      return 1
    fi

    attempt=$((attempt + 1))
    if (( attempt > NEAR_COMPLETE_RETRIES )); then
      echo "Exceeded near-complete retries for $(basename "$MODEL_ID") seed=$SEED at ${after_unique_questions}/${expected_lines} unique questions."
      return 1
    fi
    echo "Retrying near-complete $(basename "$MODEL_ID") seed=$SEED (attempt ${attempt}/${NEAR_COMPLETE_RETRIES})."
    pkill -f "run_experiment --experiment_type agent" 2>/dev/null || true
    sleep 5
  done
}

run_one_pass_with_retries "$EXPECTED_LINES"

actual_lines="$(wc -l < "$RESULT_JSONL" 2>/dev/null || echo 0)"
actual_unique_questions="$(current_unique_question_count)"
echo "Teacher raw path: $RESULT_JSONL"
echo "Teacher raw lines: $actual_lines"
echo "Teacher raw unique questions: $actual_unique_questions"

cleanup
VLLM_PID=""
