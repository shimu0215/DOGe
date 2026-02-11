#!/bin/bash
set -e

TARGET_ENV=/scratch/wzhao20/conda_envs/doge   
source ~/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_DIRS=/scratch/wzhao20/conda_envs
export CONDA_PKGS_DIRS=/scratch/wzhao20/conda_pkgs
mkdir -p /scratch/wzhao20/conda_envs /scratch/wzhao20/conda_pkgs

conda env remove -n doge -y || true
rm -rf $TARGET_ENV || true

conda create -y -p $TARGET_ENV python=3.10

conda activate $TARGET_ENV
python --version
module load cuda/12.6

NVCC_PATH=$(which nvcc || true)
if [ -z "$NVCC_PATH" ]; then
  echo "ERROR: nvcc 未找到，说明 cuda/12.6 模块可能没正确 load。"
  exit 1
fi

CUDA_DIR=$(dirname "$(dirname "$NVCC_PATH")")
export CUDA_HOME="$CUDA_DIR"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "nvcc: $NVCC_PATH"
echo "CUDA_HOME: $CUDA_HOME"

echo "=== [7] 安装所有 Python 依赖（包括 vllm） ==="
pip install -r /scratch/wzhao20/DOGe/requirements.txt

mkdir -p /scratch/wzhao20/DOGe/data

conda env list