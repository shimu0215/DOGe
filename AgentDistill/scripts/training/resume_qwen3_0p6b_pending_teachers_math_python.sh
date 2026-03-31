#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PATH="/scratch/wzhao20/conda_envs/llama-factory311-clean/bin:${PATH}"

cleanup_compute() {
  pkill -f "run_experiment --experiment_type agent" 2>/dev/null || true
  pkill -f "torchrun" 2>/dev/null || true
  pkill -f "finetune_sft.py" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "serve_vllm.py" 2>/dev/null || true
  sleep 10
}

has_complete_student_dir() {
  local dir="$1"
  [[ -f "$dir/adapter_model.bin" && -f "$dir/adapter_config.json" && -f "$dir/training_args.json" ]]
}

collect_complete_student_dirs() {
  local complete_dirs=()
  local dir
  for dir in "$@"; do
    if has_complete_student_dir "$dir"; then
      complete_dirs+=("$dir")
    else
      echo "Skipping incomplete student output during eval list build: $dir"
    fi
  done

  if (( ${#complete_dirs[@]} == 0 )); then
    return 0
  fi

  local joined=""
  local first=1
  for dir in "${complete_dirs[@]}"; do
    if (( first )); then
      joined="$dir"
      first=0
    else
      joined="$joined,$dir"
    fi
  done
  printf '%s\n' "$joined"
}

run_student_from_teacher() {
  local raw_log="$1"
  local train_tag="$2"

  cleanup_compute

  RAW_LOG="$raw_log" \
  STUDENT_MODEL="Qwen/Qwen3-0.6B" \
  TRAIN_TAG="$train_tag" \
  EPOCHS="${EPOCHS:-2}" \
  DATASET_SIZE="${DATASET_SIZE:--1}" \
  bash scripts/training/train_student_from_single_teacher_math_python.sh

  cleanup_compute
}

collect_ft_teacher() {
  local model_id="$1"
  local lora_folder="$2"

  cleanup_compute

  MODEL_ID="$model_id" \
  LORA_FOLDER="$lora_folder" \
  SEED="42" \
  N="1" \
  FORCE_RERUN="${FORCE_RERUN:-1}" \
  bash scripts/inference/collect_teacher_math_python_singletraj.sh

  cleanup_compute
}

run_if_missing() {
  local out_dir="$1"
  shift
  if has_complete_student_dir "$out_dir"; then
    echo "Skipping completed student output: $out_dir"
    return 0
  fi
  "$@"
}

cleanup_compute

BASE14B_RAW="/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test/Qwen3-14B_temp=0.7_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"
FT14B_0P2="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-14B/agent_baseline_2epochs_math14b_entropy_owntraj_ds_lambda0p2"
FT14B_0P5="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-14B/agent_baseline_2epochs_math14b_entropy_owntraj_ds_lambda0p5"
FT14B_0P8="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-14B/agent_baseline_2epochs_math14b_entropy_owntraj_ds_lambda0p8"

RAW14B_0P2="$FT14B_0P2/qa_results/math_500_20250414_test/Qwen3-14B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"
RAW14B_0P5="$FT14B_0P5/qa_results/math_500_20250414_test/Qwen3-14B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"
RAW14B_0P8="$FT14B_0P8/qa_results/math_500_20250414_test/Qwen3-14B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"

OUT14B_BASE="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math14b_base_singletraj_basicdistill"
OUT14B_0P2="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math14b_lambda0p2_singletraj_basicdistill"
OUT14B_0P5="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math14b_lambda0p5_singletraj_basicdistill"
OUT14B_0P8="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math14b_lambda0p8_singletraj_basicdistill"

# Base 14B was already completed previously; keep this guard in case of reruns.
run_if_missing "$OUT14B_BASE" run_student_from_teacher "$BASE14B_RAW" "math14b_base_singletraj_basicdistill"

if ! has_complete_student_dir "$OUT14B_0P2"; then
  collect_ft_teacher "Qwen/Qwen3-14B" "$FT14B_0P2"
  run_student_from_teacher "$RAW14B_0P2" "math14b_lambda0p2_singletraj_basicdistill"
fi

if ! has_complete_student_dir "$OUT14B_0P5"; then
  collect_ft_teacher "Qwen/Qwen3-14B" "$FT14B_0P5"
  run_student_from_teacher "$RAW14B_0P5" "math14b_lambda0p5_singletraj_basicdistill"
fi

if ! has_complete_student_dir "$OUT14B_0P8"; then
  collect_ft_teacher "Qwen/Qwen3-14B" "$FT14B_0P8"
  run_student_from_teacher "$RAW14B_0P8" "math14b_lambda0p8_singletraj_basicdistill"
fi

BASE32B_RAW="/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test/Qwen3-32B_temp=0.7_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"
FT32B_0P2="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-32B/agent_baseline_2epochs_math32b_entropy_owntraj_ds_lambda0p2"
FT32B_0P5="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-32B/agent_baseline_2epochs_math32b_entropy_owntraj_ds_lambda0p5"
FT32B_0P8="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-32B/agent_baseline_2epochs_math32b_entropy_owntraj_ds_lambda0p8"

RAW32B_0P2="$FT32B_0P2/qa_results/math_500_20250414_test/Qwen3-32B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"
RAW32B_0P5="$FT32B_0P5/qa_results/math_500_20250414_test/Qwen3-32B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"
RAW32B_0P8="$FT32B_0P8/qa_results/math_500_20250414_test/Qwen3-32B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"

OUT32B_BASE="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math32b_base_singletraj_basicdistill"
OUT32B_0P2="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math32b_lambda0p2_singletraj_basicdistill"
OUT32B_0P5="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math32b_lambda0p5_singletraj_basicdistill"
OUT32B_0P8="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math32b_lambda0p8_singletraj_basicdistill"

run_if_missing "$OUT32B_BASE" run_student_from_teacher "$BASE32B_RAW" "math32b_base_singletraj_basicdistill"

if ! has_complete_student_dir "$OUT32B_0P2"; then
  collect_ft_teacher "Qwen/Qwen3-32B" "$FT32B_0P2"
  run_student_from_teacher "$RAW32B_0P2" "math32b_lambda0p2_singletraj_basicdistill"
fi

if ! has_complete_student_dir "$OUT32B_0P5"; then
  collect_ft_teacher "Qwen/Qwen3-32B" "$FT32B_0P5"
  run_student_from_teacher "$RAW32B_0P5" "math32b_lambda0p5_singletraj_basicdistill"
fi

if ! has_complete_student_dir "$OUT32B_0P8"; then
  collect_ft_teacher "Qwen/Qwen3-32B" "$FT32B_0P8"
  run_student_from_teacher "$RAW32B_0P8" "math32b_lambda0p8_singletraj_basicdistill"
fi

cleanup_compute

EVAL_LORA_FOLDERS="$(collect_complete_student_dirs \
  "$OUT14B_BASE" "$OUT14B_0P2" "$OUT14B_0P5" "$OUT14B_0P8" \
  "$OUT32B_BASE" "$OUT32B_0P2" "$OUT32B_0P5" "$OUT32B_0P8")"

if [[ -n "$EVAL_LORA_FOLDERS" ]]; then
  LORA_FOLDERS="$EVAL_LORA_FOLDERS" \
  bash scripts/inference/eval_qwen3_0p6b_math_python_students.sh
else
  echo "No complete student outputs available for evaluation yet."
fi

cleanup_compute
