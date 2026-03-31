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

RAW_ROOT="${RAW_ROOT:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test}"
TARGET_MODEL="Qwen/Qwen3-32B"
TRAIN_TAG_PREFIX="${TRAIN_TAG_PREFIX:-math32b_entropy_self32}"
EPOCHS="${EPOCHS:-2}"
LAMBDAS=(${LAMBDAS:-0.2})
MAX_LENGTH="${MAX_LENGTH:-1024}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"

mkdir -p logs

RAW_LOGS=()
for seed in $(seq 42 56); do
  raw_log="${RAW_ROOT}/Qwen3-32B_temp=0.7_seed=${seed}_type=agent_steps=5_python_only_python_only_seed${seed}.jsonl"
  if [[ ! -f "$raw_log" ]]; then
    echo "Missing raw teacher log: $raw_log" >&2
    exit 1
  fi
  RAW_LOGS+=("$raw_log")
done

FILTERED_LOGS=()
for raw_log in "${RAW_LOGS[@]}"; do
  python -m exps_research.unified_framework.score_answers \
    --log_files "$raw_log" \
    --task_type math \
    --max_workers 8

  scored_log="$(dirname "$raw_log")/evaluations/$(basename "${raw_log%.jsonl}")_scored.jsonl"
  if [[ ! -f "$scored_log" ]]; then
    echo "Scored log not found: $scored_log" >&2
    exit 1
  fi

  python -m exps_research.unified_framework.filter_agent_training_data \
    --result_path "$scored_log" \
    --do_save

  filtered_log="$(dirname "$raw_log")/filtered_data/$(basename "${raw_log%.jsonl}")_filtered.jsonl"
  if [[ ! -f "$filtered_log" ]]; then
    echo "Filtered log not found: $filtered_log" >&2
    exit 1
  fi

  FILTERED_LOGS+=("$filtered_log")
done

for lambda in "${LAMBDAS[@]}"; do
  lambda_tag="${lambda/./p}"
  postfix="${TRAIN_TAG_PREFIX}_lambda${lambda_tag}"
  run_log="logs/train_teacher_entropy_32b_lambda${lambda_tag}.log"
  echo "=== Training $TARGET_MODEL with entropy lambda=$lambda ===" | tee "$run_log"
  torchrun --nproc_per_node=4 exps_research/finetune_sft.py \
    --model_name "$TARGET_MODEL" \
    --num_epochs "$EPOCHS" \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr 2e-4 \
    --train_filepath "${FILTERED_LOGS[@]}" \
    --postfix "$postfix" \
    --solution_type agent \
    --fsdp exps_research/mp_configs/fsdp.json \
    --max_length "$MAX_LENGTH" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --random_trajectory_per_question \
    --use_entropy_regularization \
    --entropy_lambda "$lambda" \
    2>&1 | tee -a "$run_log"
done
