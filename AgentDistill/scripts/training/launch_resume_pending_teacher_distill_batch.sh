#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PATH="/scratch/wzhao20/conda_envs/llama-factory311-clean/bin:${PATH}"

pkill -f "run_experiment --experiment_type agent" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
pkill -f "finetune_sft.py" 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "serve_vllm.py" 2>/dev/null || true
pkill -f "run_qwen3_0p6b_from_14b_teachers_math_python.sh" 2>/dev/null || true
pkill -f "run_qwen3_0p6b_from_32b_teachers_math_python.sh" 2>/dev/null || true
pkill -f "resume_qwen3_0p6b_pending_teachers_math_python.sh" 2>/dev/null || true
sleep 10

bash scripts/training/resume_qwen3_0p6b_pending_teachers_math_python.sh
