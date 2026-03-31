#!/bin/bash
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --time=02:00:00
#SBATCH --job-name=iter-a100-2h-cq
#SBATCH --output=hpc-results/iter-a100-2h-cq-%j.out
#SBATCH --error=hpc-results/iter-a100-2h-cq-%j.err

set -euo pipefail

mkdir -p hpc-results

hostname
nvidia-smi
sleep 7200
