#!/bin/bash
set -euo pipefail

MODULAR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${MODULAR_DIR}/.." && pwd)}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/wzhao20/conda_envs/AKDA1}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${CONDA_ENV_PREFIX}/bin/vllm}"

setup_agentdistill_env() {
  cd "$ROOT_DIR"

  export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"
  export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
  export HF_HOME="${HF_HOME:-/scratch/wzhao20/hf_cache}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/wzhao20/.cache}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/wzhao20/vllm_cache}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/wzhao20/triton_cache}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/wzhao20/torchinductor_cache}"
  export VLLM_NO_USAGE_STATS=1
  export DO_NOT_TRACK=1
}

cleanup_collection_resources() {
  pkill -f "run_experiment" 2>/dev/null || true
  pkill -f "serve_vllm.py" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
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

expected_question_count() {
  local data_path="$1"
  "$PYTHON_BIN" - "$data_path" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
if isinstance(data, dict) and "examples" in data:
    print(len(data["examples"]))
else:
    print(len(data))
PY
}

result_jsonl_path() {
  local model_id="$1"
  local data_path="$2"
  local seed="$3"
  local max_steps="$4"
  local n="$5"
  local lora_folder="${6:-}"
  local log_root="${7:-/scratch/wzhao20/AKDA2/AgentDistill/logs/qa_results_python_only_teacher}"
  local name_tag="${8:-}"
  local temperature="${9:-0.7}"
  local model_name dataset_name base_dir

  model_name="$(basename "$model_id")"
  if [[ -n "$name_tag" ]]; then
    dataset_name="$name_tag"
  else
    dataset_name="$(basename "$data_path" .json)"
  fi

  if [[ -n "$lora_folder" ]]; then
    base_dir="${lora_folder}/qa_results"
    printf "%s/%s_test/%s_%s_temp=%s_n=%s_seed=%s_type=agent_steps=%s_python_only_python_only_seed%s.jsonl" \
      "$base_dir" "$dataset_name" "$model_name" "$dataset_name" "$temperature" "$n" "$seed" "$max_steps" "$seed"
  else
    base_dir="$log_root"
    printf "%s/%s_test/%s_%s_temp=%s_seed=%s_type=agent_steps=%s_python_only_python_only_seed%s.jsonl" \
      "$base_dir" "$dataset_name" "$model_name" "$dataset_name" "$temperature" "$seed" "$max_steps" "$seed"
  fi
}

count_unique_questions() {
  local result_path="$1"
  "$PYTHON_BIN" - "$result_path" <<'PY'
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

is_collection_complete() {
  local result_path="$1"
  local expected_count="$2"
  [[ -f "$result_path" ]] || return 1
  local unique_count
  unique_count="$(count_unique_questions "$result_path")"
  [[ "$unique_count" -ge "$expected_count" ]]
}

build_remaining_dataset() {
  local source_data_path="$1"
  local existing_raw="$2"
  local output_json="$3"
  "$PYTHON_BIN" - "$source_data_path" "$existing_raw" "$output_json" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
existing_path = Path(sys.argv[2])
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
if existing_path.exists():
    with existing_path.open() as f:
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

merge_raw_results_by_question() {
  local primary_path="$1"
  local extra_path="$2"
  local output_path="$3"
  "$PYTHON_BIN" - "$primary_path" "$extra_path" "$output_path" <<'PY'
import json
import sys
from pathlib import Path

primary = Path(sys.argv[1])
extra = Path(sys.argv[2])
output = Path(sys.argv[3])

merged = {}
order = []

def load(path):
    if not path.exists():
        return
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
            if not question:
                continue
            if question not in merged:
                order.append(question)
            merged[question] = entry

load(primary)
load(extra)

with output.open("w") as f:
    for question in order:
        f.write(json.dumps(merged[question], ensure_ascii=False) + "\n")
print(len(order))
PY
}

infer_task_type() {
  local data_path="$1"
  local lower
  lower="$(echo "$data_path" | tr '[:upper:]' '[:lower:]')"

  if [[ "$lower" == *"math"* || "$lower" == *"gsm"* || "$lower" == *"aime"* || "$lower" == *"olym"* ]]; then
    echo "math"
  else
    echo "fact"
  fi
}
