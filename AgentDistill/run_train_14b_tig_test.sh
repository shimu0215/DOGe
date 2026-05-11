#!/bin/bash
set -e

cd /scratch/wzhao20/AKDA2/AgentDistill

export PATH=/scratch/wzhao20/conda_envs/AKDA1/bin:$PATH
export HF_HOME=/scratch/wzhao20/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME
export TRITON_CACHE_DIR=/scratch/wzhao20/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/wzhao20/torchinductor_cache
export VLLM_CACHE_ROOT=/scratch/wzhao20/vllm_cache
export XDG_CACHE_HOME=/scratch/wzhao20/.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=2,3 /scratch/wzhao20/conda_envs/AKDA1/bin/accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  -m exps_online_rl.codeact_grpo \
  --model_name Qwen/Qwen3-14B \
  --data_path /scratch/wzhao20/AKDA2/AgentDistill/data_processor/math_dataset/test/math_500_20250414.json \
  --output_root /scratch/wzhao20/AKDA2/AgentDistill/training_outputs/qwen3-14B/online_grpo_math500_tig_test \
  --seed 42 \
  --num_questions_per_sync 2 \
  --num_rollouts_per_question 4 \
  --max_syncs 3 \
  --grpo_epochs 1 \
  --train_batch_size 1 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-6 \
  --temperature 0.7 \
  --save_every_syncs 999 \
  --rollout_workers 4 \
  --rollout_timeout_seconds 600 \
  --rollout_port 8000 \
  --rollout_tp 2 \
  --rollout_gpu_util 0.9 \
  --rollout_cuda_visible_devices 0,1 \
  --rollout_max_length 32768 \
  --max_steps 6 \
  --max_length 8192 \
  --max_tokens 1024 \
  --lora_r 32 \
  --lora_alpha 64 \
  --use_per_token_ratio 1 \
  --use_dual_clip 1 \
  --dual_clip_delta 3.0 \
  --use_kl_loss 0 \
  --use_log_ratio_clip 0 \
  --use_tig_reward 1 \
  --tig_lambda 0.5 \
  --tig_eval_batch_size 2 \
  --python_bin /scratch/wzhao20/conda_envs/AKDA1/bin/python \
  --distributed_timeout_minutes 240
