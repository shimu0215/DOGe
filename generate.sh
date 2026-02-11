# export PYTHONPATH=$PYTHONPATH:src

# generate distillation data for R1-Qwen-7B and Qwen3-8B models with gsm8k dataset
# python scripts/generate-distillation-data.py \
#    --dataset_name="openai/gsm8k" --save_dir="data/r1-qwen-7b-gsm8k/" \
#    --model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --max_tokens=8192

# python scripts/generate-distillation-data.py \
#   --dataset_name="openai/gsm8k" --save_dir="data/qwen3-8b-gsm8k-v2/" \
#   --model_name="Qwen/Qwen3-8B" --base_url="http://localhost:2333/v1" --max_tokens=8192


# generate distillation data for R1-Qwen-7B and Qwen3-8B models with general/mixed dataset
# python scripts/generate-distillation-data.py \
#   --dataset_name="ServiceNow-AI/R1-Distill-SFT" --save_dir="data/r1-qwen-7b-mixed/" \
#   --model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --max_tokens=8192

# python scripts/generate-distillation-data.py \
#   --dataset_name="ServiceNow-AI/R1-Distill-SFT" --save_dir="data/qwen3-8b-mixed-v2/" \
#   --model_name="Qwen/Qwen3-8B" --base_url="http://localhost:2333/v1" --max_tokens=8192
  
export PYTHONPATH=$PYTHONPATH:src

# 用 Qwen2.5-7B-Instruct 作为 teacher，在 GSM8K 上生成蒸馏数据
  
  python scripts/generate-distillation-data.py \
  --dataset_name="reasoning-machines/gsm-hard" \
  --save_dir="data/qwen2_5-7b-gsm-hard/" \
  --model_name="Qwen/Qwen2.5-7B-Instruct" \
  --base_url="http://127.0.0.1:2333/v1" \
  --max_tokens=8192