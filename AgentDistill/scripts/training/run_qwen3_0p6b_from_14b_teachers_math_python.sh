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

run_one_teacher() {
  local teacher_kind="$1"
  local raw_log="$2"
  local train_tag="$3"

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
  local lora_folder="$1"

  cleanup_compute

  MODEL_ID="Qwen/Qwen3-14B" \
  LORA_FOLDER="$lora_folder" \
  SEED="42" \
  N="1" \
  FORCE_RERUN="${FORCE_RERUN:-1}" \
  bash scripts/inference/collect_teacher_math_python_singletraj.sh

  cleanup_compute
}

BASE_RAW="/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test/Qwen3-14B_temp=0.7_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"
FT14B_LAMBDA0P2="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-14B/agent_baseline_2epochs_math14b_entropy_owntraj_ds_lambda0p2"
FT14B_LAMBDA0P5="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-14B/agent_baseline_2epochs_math14b_entropy_owntraj_ds_lambda0p5"
FT14B_LAMBDA0P8="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-14B/agent_baseline_2epochs_math14b_entropy_owntraj_ds_lambda0p8"

FT_RAW_0P2="$FT14B_LAMBDA0P2/qa_results/math_500_20250414_test/Qwen3-14B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"
FT_RAW_0P5="$FT14B_LAMBDA0P5/qa_results/math_500_20250414_test/Qwen3-14B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"
FT_RAW_0P8="$FT14B_LAMBDA0P8/qa_results/math_500_20250414_test/Qwen3-14B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl"

cleanup_compute

run_one_teacher "base14b" "$BASE_RAW" "math14b_base_singletraj_basicdistill"

collect_ft_teacher "$FT14B_LAMBDA0P2"
run_one_teacher "14b_lambda0p2" "$FT_RAW_0P2" "math14b_lambda0p2_singletraj_basicdistill"

collect_ft_teacher "$FT14B_LAMBDA0P5"
run_one_teacher "14b_lambda0p5" "$FT_RAW_0P5" "math14b_lambda0p5_singletraj_basicdistill"

collect_ft_teacher "$FT14B_LAMBDA0P8"
run_one_teacher "14b_lambda0p8" "$FT_RAW_0P8" "math14b_lambda0p8_singletraj_basicdistill"

cleanup_compute

LORA_FOLDERS="/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math14b_base_singletraj_basicdistill,/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math14b_lambda0p2_singletraj_basicdistill,/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math14b_lambda0p5_singletraj_basicdistill,/scratch/wzhao20/AgentDistill/training_outputs/qwen3-0.6B/agent_baseline_2epochs_math14b_lambda0p8_singletraj_basicdistill" \
bash scripts/inference/eval_qwen3_0p6b_math_python_students.sh

cleanup_compute
