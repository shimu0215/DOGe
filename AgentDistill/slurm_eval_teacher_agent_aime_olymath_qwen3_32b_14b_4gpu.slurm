#!/bin/bash
#SBATCH --job-name=ad-aime-olym-32b14b
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs_slurm/%x-%j.out
#SBATCH --error=logs_slurm/%x-%j.err

set -euo pipefail

ROOT_DIR="/scratch/wzhao20/AgentDistill"
CONDA_ENV="/scratch/wzhao20/conda_envs/llama-factory311-clean"

mkdir -p "${ROOT_DIR}/logs_slurm"

set +u
source /home/wzhao20/miniconda3/etc/profile.d/conda.sh
set -u
conda activate "${CONDA_ENV}"

cd "${ROOT_DIR}"

export HF_HOME="/scratch/wzhao20/hf_cache"
export TRANSFORMERS_CACHE="/scratch/wzhao20/hf_cache"
export HF_DATASETS_CACHE="/scratch/wzhao20/hf_cache/datasets"
export XDG_CACHE_HOME="/scratch/wzhao20/.cache"
export VLLM_CACHE_ROOT="/scratch/wzhao20/vllm_cache"
export TRITON_CACHE_DIR="/scratch/wzhao20/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/scratch/wzhao20/torchinductor_cache"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

export DATASET_LIST="/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/aime_90_20250504.json,/scratch/wzhao20/AgentDistill/data_processor/math_dataset/test/olymath_200_20250511.json"
export MODEL_LIST="Qwen/Qwen3-32B,Qwen/Qwen3-14B"

echo "[info] starting teacher data sequence for AIME and OlymMATH"
echo "[info] datasets: ${DATASET_LIST}"
echo "[info] models: ${MODEL_LIST}"

bash scripts/inference/run_agent_python_only_data_sequence.sh
