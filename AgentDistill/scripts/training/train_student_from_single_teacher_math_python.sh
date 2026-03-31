#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/wzhao20/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/wzhao20/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/wzhao20/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/wzhao20/torchinductor_cache}"

RAW_LOG="${RAW_LOG:?RAW_LOG is required}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-0.6B}"
TRAIN_TAG="${TRAIN_TAG:-math_teacher_singletraj_basicdistill}"
EPOCHS="${EPOCHS:-1}"
DATASET_SIZE="${DATASET_SIZE:-128}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-exps_research/mp_configs/ds3_no_offload.json}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-5}"

mkdir -p logs

monitor_system_usage() {
  local monitor_log="$1"
  local target_pid="$2"

  (
    while kill -0 "$target_pid" 2>/dev/null; do
      {
        echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
        free -h
        echo "__TOP_PYTHON_RSS_KB__"
        ps -eo pid,ppid,rss,vsz,comm,args --sort=-rss | grep -E 'python|torchrun|deepspeed' | grep -v grep | head -n 12 || true
        echo "__NVIDIA_SMI__"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
        echo
      } >> "$monitor_log"
      sleep "$MONITOR_INTERVAL_SECONDS"
    done
  ) &
  echo $!
}

count_lines() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -l < "$path"
  else
    echo 0
  fi
}

if [[ ! -f "$RAW_LOG" ]]; then
  echo "Raw teacher log not found: $RAW_LOG" >&2
  exit 1
fi

python -m exps_research.unified_framework.score_answers \
  --log_files "$RAW_LOG" \
  --task_type math \
  --max_workers 8

SCORED_LOG="$(dirname "$RAW_LOG")/evaluations/$(basename "${RAW_LOG%.jsonl}")_scored.jsonl"
if [[ ! -f "$SCORED_LOG" ]]; then
  echo "Scored log not found: $SCORED_LOG" >&2
  exit 1
fi

python -m exps_research.unified_framework.filter_agent_training_data \
  --result_path "$SCORED_LOG" \
  --do_save

FILTERED_LOG="$(dirname "$RAW_LOG")/filtered_data/$(basename "${RAW_LOG%.jsonl}")_filtered.jsonl"
if [[ ! -f "$FILTERED_LOG" ]]; then
  echo "Filtered log not found: $FILTERED_LOG" >&2
  exit 1
fi

raw_count="$(count_lines "$RAW_LOG")"
scored_count="$(count_lines "$SCORED_LOG")"
filtered_count="$(count_lines "$FILTERED_LOG")"
echo "=== Student distillation filtering summary ==="
printf 'raw=%s scored=%s filtered=%s\n' "$raw_count" "$scored_count" "$filtered_count"

python - "$FILTERED_LOG" <<'PY'
import json
import sys
path = sys.argv[1]
questions = set()
rows = 0
with open(path) as f:
    for line in f:
        rows += 1
        row = json.loads(line)
        q = row.get("question")
        if q is not None:
            questions.add(q)
print(f"unique_questions={len(questions)}")
print(f"filtered_rows={rows}")
PY

postfix="$TRAIN_TAG"
run_log="logs/train_student_${TRAIN_TAG}.log"
monitor_log="logs/train_student_${TRAIN_TAG}_monitor.log"

echo "=== Training $STUDENT_MODEL from $FILTERED_LOG ===" | tee "$run_log"
echo "Using DeepSpeed config: $DEEPSPEED_CONFIG" | tee -a "$run_log"
echo "Using one filtered teacher trajectory per question source file; no grouped sampling or voting." | tee -a "$run_log"
echo "EPOCHS=$EPOCHS DATASET_SIZE=$DATASET_SIZE MAX_LENGTH=$MAX_LENGTH LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA" | tee -a "$run_log"

torchrun --nproc_per_node=4 exps_research/finetune_sft.py \
  --model_name "$STUDENT_MODEL" \
  --num_epochs "$EPOCHS" \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lr 2e-4 \
  --train_filepath "$FILTERED_LOG" \
  --postfix "$postfix" \
  --solution_type agent \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --gradient_checkpointing \
  --max_length "$MAX_LENGTH" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --dataset_size "$DATASET_SIZE" \
  >> "$run_log" 2>&1 &
train_pid=$!

monitor_pid="$(monitor_system_usage "$monitor_log" "$train_pid")"

set +e
wait "$train_pid"
train_status=$?
set -e

if kill -0 "$monitor_pid" 2>/dev/null; then
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
fi

if [[ "$train_status" -ne 0 ]]; then
  echo "Student distillation run failed with exit code $train_status. See $run_log and $monitor_log" | tee -a "$run_log"
  exit "$train_status"
fi
