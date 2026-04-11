#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/wzhao20/conda_envs/AKDA1}"
export PATH="$CONDA_ENV_PREFIX/bin:${PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_PREFIX/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$CONDA_ENV_PREFIX/bin/torchrun}"
export HF_HOME="${HF_HOME:-/scratch/wzhao20/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/wzhao20/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/wzhao20/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/wzhao20/torchinductor_cache}"

RAW_ROOT="${RAW_ROOT:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-14B}"
RAW_MODEL_NAME="${RAW_MODEL_NAME:-Qwen3-14B}"
TRAIN_TAG_PREFIX="${TRAIN_TAG_PREFIX:-math14b_entropy_owntraj_ds}"
EPOCHS="${EPOCHS:-2}"
LAMBDAS=(${LAMBDAS:-0.2 0.5 0.8})
SEED_START="${SEED_START:-42}"
SEED_END="${SEED_END:-56}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-exps_research/mp_configs/ds3_no_offload.json}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-5}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
ENTROPY_ON_THOUGHT_ONLY="${ENTROPY_ON_THOUGHT_ONLY:-0}"
EARLY_STOP_PATIENCE_EPOCHS="${EARLY_STOP_PATIENCE_EPOCHS:-0}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0}"

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

RAW_LOGS=()
FILTERED_LOGS=()

for seed in $(seq "$SEED_START" "$SEED_END"); do
  raw_log="${RAW_ROOT}/${RAW_MODEL_NAME}_temp=0.7_seed=${seed}_type=agent_steps=5_python_only_python_only_seed${seed}.jsonl"
  if [[ ! -f "$raw_log" ]]; then
    echo "Missing raw teacher log: $raw_log" >&2
    exit 1
  fi
  RAW_LOGS+=("$raw_log")

  "$PYTHON_BIN" -m exps_research.unified_framework.score_answers \
    --log_files "$raw_log" \
    --task_type math \
    --max_workers 8

  scored_log="$(dirname "$raw_log")/evaluations/$(basename "${raw_log%.jsonl}")_scored.jsonl"
  if [[ ! -f "$scored_log" ]]; then
    echo "Scored log not found: $scored_log" >&2
    exit 1
  fi

  "$PYTHON_BIN" -m exps_research.unified_framework.filter_agent_training_data \
    --result_path "$scored_log" \
    --do_save

  filtered_log="$(dirname "$raw_log")/filtered_data/$(basename "${raw_log%.jsonl}")_filtered.jsonl"
  if [[ ! -f "$filtered_log" ]]; then
    echo "Filtered log not found: $filtered_log" >&2
    exit 1
  fi

  FILTERED_LOGS+=("$filtered_log")
done

echo "=== 14B own-data filtering summary before training ==="
total_raw=0
total_scored=0
total_filtered=0
for seed in $(seq "$SEED_START" "$SEED_END"); do
  raw_log="${RAW_ROOT}/${RAW_MODEL_NAME}_temp=0.7_seed=${seed}_type=agent_steps=5_python_only_python_only_seed${seed}.jsonl"
  scored_log="$(dirname "$raw_log")/evaluations/$(basename "${raw_log%.jsonl}")_scored.jsonl"
  filtered_log="$(dirname "$raw_log")/filtered_data/$(basename "${raw_log%.jsonl}")_filtered.jsonl"
  raw_count="$(count_lines "$raw_log")"
  scored_count="$(count_lines "$scored_log")"
  filtered_count="$(count_lines "$filtered_log")"
  total_raw=$((total_raw + raw_count))
  total_scored=$((total_scored + scored_count))
  total_filtered=$((total_filtered + filtered_count))
  printf 'seed=%s raw=%s scored=%s filtered=%s\n' "$seed" "$raw_count" "$scored_count" "$filtered_count"
done
printf 'total raw=%s scored=%s filtered=%s\n' "$total_raw" "$total_scored" "$total_filtered"

"$PYTHON_BIN" - "${FILTERED_LOGS[@]}" <<'PY'
import json
import sys

paths = sys.argv[1:]
unique_questions = set()
total_rows = 0
for path in paths:
    with open(path) as f:
        for line in f:
            total_rows += 1
            row = json.loads(line)
            q = row.get("question")
            if q is not None:
                unique_questions.add(q)

print(f"unique_questions={len(unique_questions)}")
print(f"filtered_rows={total_rows}")
PY

for lambda in "${LAMBDAS[@]}"; do
  lambda_tag="${lambda/./p}"
  postfix="${TRAIN_TAG_PREFIX}_lambda${lambda_tag}"
  run_log="logs/train_teacher_entropy_14b_owntraj_deepspeed_lambda${lambda_tag}.log"
  monitor_log="logs/train_teacher_entropy_14b_owntraj_deepspeed_lambda${lambda_tag}_monitor.log"

  echo "=== Training $TARGET_MODEL with entropy lambda=$lambda via DeepSpeed on 14B own trajectories ===" | tee "$run_log"
  echo "Using DeepSpeed config: $DEEPSPEED_CONFIG" | tee -a "$run_log"
  echo "Using all filtered trajectories as independent SFT samples." | tee -a "$run_log"
  echo "EPOCHS=$EPOCHS MAX_LENGTH=$MAX_LENGTH LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA" | tee -a "$run_log"
  echo "SAVE_STRATEGY=$SAVE_STRATEGY SAVE_STEPS=$SAVE_STEPS SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT" | tee -a "$run_log"
  echo "ENTROPY_ON_THOUGHT_ONLY=$ENTROPY_ON_THOUGHT_ONLY" | tee -a "$run_log"
  echo "EARLY_STOP_PATIENCE_EPOCHS=$EARLY_STOP_PATIENCE_EPOCHS EARLY_STOP_MIN_DELTA=$EARLY_STOP_MIN_DELTA" | tee -a "$run_log"

  train_cmd=(
    "$TORCHRUN_BIN" --nproc_per_node="${NPROC_PER_NODE:-4}" exps_research/finetune_sft.py
    --model_name "$TARGET_MODEL"
    --num_epochs "$EPOCHS"
    --batch_size 1
    --gradient_accumulation_steps 4
    --lr 2e-4
    --train_filepath "${FILTERED_LOGS[@]}"
    --postfix "$postfix"
    --solution_type agent
    --deepspeed "$DEEPSPEED_CONFIG"
    --gradient_checkpointing
    --max_length "$MAX_LENGTH"
    --save_strategy "$SAVE_STRATEGY"
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --early_stop_patience_epochs "$EARLY_STOP_PATIENCE_EPOCHS"
    --early_stop_min_delta "$EARLY_STOP_MIN_DELTA"
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --use_entropy_regularization
    --entropy_lambda "$lambda"
  )
  if [[ "$ENTROPY_ON_THOUGHT_ONLY" == "1" ]]; then
    train_cmd+=(--entropy_on_thought_only)
  fi

  "${train_cmd[@]}" >> "$run_log" 2>&1 &
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
    echo "14B own-data DeepSpeed run failed with exit code $train_status. See $run_log and $monitor_log" | tee -a "$run_log"
    exit "$train_status"
  fi
done
