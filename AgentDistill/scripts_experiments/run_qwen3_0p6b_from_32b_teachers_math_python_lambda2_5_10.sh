#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/wzhao20/conda_envs/AKDA1}"
export PATH="$CONDA_ENV_PREFIX/bin:${PATH:-}"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

cleanup_compute() {
  pkill -f "run_experiment --experiment_type agent" 2>/dev/null || true
  pkill -f "torchrun" 2>/dev/null || true
  pkill -f "finetune_sft.py" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "serve_vllm.py" 2>/dev/null || true
  sleep 10
}

raw_unique_count() {
  local raw_path="$1"
  "$CONDA_ENV_PREFIX/bin/python" - "$raw_path" <<'PY'
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
            obj = json.loads(line)
        except Exception:
            continue
        q = obj.get("question") or obj.get("problem") or obj.get("prompt")
        if q:
            seen.add(q)
print(len(seen))
PY
}

merge_tmp_teacher_raws() {
  local teacher_dir="$1"
  local raw_log="$2"
  local merged_tmp="${raw_log}.merged.$$"
  "$CONDA_ENV_PREFIX/bin/python" - "$teacher_dir" "$raw_log" "$merged_tmp" <<'PY'
import json
import sys
from pathlib import Path

teacher_dir = Path(sys.argv[1])
raw_log = Path(sys.argv[2])
merged_out = Path(sys.argv[3])
qa_root = teacher_dir / "qa_results"

paths = []
if raw_log.exists():
    paths.append(raw_log)
for p in sorted(qa_root.glob("teacher_collect_remaining.*_tmp/*.jsonl")):
    paths.append(p)

merged = {}
order = []
for p in paths:
    if not p.exists():
        continue
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            q = obj.get("question") or obj.get("problem") or obj.get("prompt")
            if not q:
                continue
            if q not in merged:
                order.append(q)
            merged[q] = obj

merged_out.parent.mkdir(parents=True, exist_ok=True)
with merged_out.open("w") as f:
    for q in order:
        f.write(json.dumps(merged[q], ensure_ascii=False) + "\n")
print(len(order))
PY

  if [[ -f "$merged_tmp" ]]; then
    mv "$merged_tmp" "$raw_log"
  fi
}

ensure_teacher_raw() {
  local teacher_dir="$1"
  local raw_log="$2"

  merge_tmp_teacher_raws "$teacher_dir" "$raw_log"
  local unique_count
  unique_count="$(raw_unique_count "$raw_log")"
  if [[ "$unique_count" -ge 500 ]]; then
    return 0
  fi

  collect_ft_teacher "$teacher_dir"
  merge_tmp_teacher_raws "$teacher_dir" "$raw_log"
  unique_count="$(raw_unique_count "$raw_log")"
  if [[ "$unique_count" -lt 500 ]]; then
    echo "Teacher raw still incomplete after collection: $raw_log ($unique_count/500 unique)" >&2
    return 1
  fi
}

run_one_teacher() {
  local raw_log="$1"
  local train_tag="$2"

  cleanup_compute

  RAW_LOG="$raw_log" \
  STUDENT_MODEL="Qwen/Qwen3-0.6B" \
  TRAIN_TAG="$train_tag" \
  EPOCHS="${EPOCHS:-2}" \
  DATASET_SIZE="${DATASET_SIZE:--1}" \
  bash scripts_modular/train_student_from_single_teacher_math_python.sh

  cleanup_compute
}

collect_ft_teacher() {
  local lora_folder="$1"

  cleanup_compute

  MODEL_ID="Qwen/Qwen3-32B" \
  LORA_FOLDER="$lora_folder" \
  SEED="42" \
  N="1" \
  FORCE_RERUN="${FORCE_RERUN:-0}" \
  bash scripts/inference/collect_teacher_math_python_singletraj.sh

  cleanup_compute
}

for lambda in 2 5 10; do
  lambda_tag="${lambda/./p}"
  teacher_dir="/scratch/wzhao20/AKDA2/AgentDistill/training_outputs/qwen3-32B/agent_baseline_2epochs_agent_baseline_2epochs_math32b_entropy_owntraj_ds_lambda${lambda_tag}"
  raw_log="$teacher_dir/qa_results/math_500_20250414_test/Qwen3-32B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"

  ensure_teacher_raw "$teacher_dir" "$raw_log"
  run_one_teacher "$raw_log" "math32b_lambda${lambda_tag}_singletraj_basicdistill"
done

cleanup_compute
