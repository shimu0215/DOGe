#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --job-name=iter-a100-4g-4h-full-gq
#SBATCH --output=hpc-results/iter-a100-4g-4h-full-gq-%j.out
#SBATCH --error=hpc-results/iter-a100-4g-4h-full-gq-%j.err

set -euo pipefail

mkdir -p hpc-results

hostname
nvidia-smi
sleep 14400
