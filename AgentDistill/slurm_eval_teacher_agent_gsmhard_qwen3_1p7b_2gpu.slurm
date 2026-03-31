#!/bin/bash
#SBATCH --job-name=ad-gsm-qwen3-1p7b
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=04:00:00
#SBATCH --output=logs_slurm/%x-%j.out
#SBATCH --error=logs_slurm/%x-%j.err

set -euo pipefail

# ===== User settings =====
ROOT_DIR="/scratch/wzhao20/AgentDistill"
CONDA_ENV="/scratch/wzhao20/conda_envs/agents"
MODEL_ID="Qwen/Qwen3-1.7B"
DATA_PATH="data_processor/math_dataset/test/gsm_hard_500_20250507.json"
PORT=8000
MAX_TOKENS=1024
# =========================

mkdir -p logs_slurm

# Some clusters' /etc/bashrc assumes unset vars may exist; avoid nounset crash while sourcing.
set +u
source ~/.bashrc
set -u
module load cuda/12.6 || true
conda activate "${CONDA_ENV}"
cd "${ROOT_DIR}"

PIDS=()
cleanup() {
  echo "[cleanup] stop background services"
  kill "${PIDS[@]:-}" 2>/dev/null || true
  ps -u "$USER" -o pid,command | grep 'vllm serve' | grep -v grep | awk '{print $1}' | xargs -r kill || true
}
trap 'cleanup' EXIT INT TERM

VLLM_LOG="logs_slurm/vllm_qwen3_1p7b_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=0,1 python serve_vllm.py \
  --model "${MODEL_ID}" \
  --tensor-parallel-size 2 \
  --port "${PORT}" \
  > "${VLLM_LOG}" 2>&1 &
PIDS+=($!)

echo "[wait] waiting for vLLM startup..."
for _ in $(seq 1 180); do
  if grep -q "Application startup complete" "${VLLM_LOG}"; then
    echo "[ok] vLLM is ready"
    break
  fi
  sleep 2
done

if ! grep -q "Application startup complete" "${VLLM_LOG}"; then
  echo "[error] vLLM did not start in time"
  exit 1
fi

# Teacher-style agent evaluation, simplest setup:
# - agent mode
# - no prefix/cot memory
# - n=1 (no majority voting)
# - temperature=0.0
python -m exps_research.unified_framework.run_experiment \
  --experiment_type agent \
  --data_path "${DATA_PATH}" \
  --model_type vllm \
  --model_id "${MODEL_ID}" \
  --search_engine_type duckduckgo \
  --max_steps 5 \
  --max_tokens "${MAX_TOKENS}" \
  --n 1 \
  --temperature 0.0 \
  --top_p 0.8 \
  --seed 42 \
  --verbose \
  --use_single_endpoint
