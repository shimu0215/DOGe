#!/bin/bash
#SBATCH --job-name=akda-evalDogeteacher
#SBATCH --qos=gpu  # or gpu/cs_dept
#SBATCH --partition=gpuq   # contrib-gpuq or gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.40gb:4
#SBATCH --output=akda-evalDogeteacher-out.out   # Output file
#SBATCH --error=akda-evalDogeteacher-error.err    # Error file
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=12:00:00

# keep whatever you want
source ~/.bashrc
export HF_HOME=/scratch/wzhao20/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=/scratch/wzhao20/vllm_cache
cd /scratch/wzhao20/agent-distillation/agent-distillation
module load cuda/12.6
conda activate /home/wzhao20/miniconda3/envs/agents

pkill -f "vllm serve" || true
pkill -f "VllmWorker" || true
pkill -f "EngineCore" || true

unset RANK WORLD_SIZE LOCAL_RANK NODE_RANK MASTER_ADDR MASTER_PORT

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 1000))

IFNAME=$(ip route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
echo "Using interface: $IFNAME, MASTER_PORT=$MASTER_PORT"

export GLOO_SOCKET_IFNAME=$IFNAME
export NCCL_SOCKET_IFNAME=$IFNAME
export GLOO_USE_IPV6=0

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# nsys profile --delay=120 --duration=300 --output=dp16 --trace=cuda,osrt \
#  --sampling-period=28000000 --force-overwrite=true --stats=true \
#  ./examples/llama/train_llama3_1b_a100_dp16.sh
# python xxx.py
# srun xxx.py
# bash /scratch/wzhao20/agent-distillation/agent-distillation/scripts/training/train_agent.sh Qwen/Qwen2.5-1.5B-Instruct
bash /scratch/wzhao20/agent-distillation/agent-distillation/scripts/inference/run_agent_teacher_test.sh
