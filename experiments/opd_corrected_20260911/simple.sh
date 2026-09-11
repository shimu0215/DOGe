#!/usr/bin/env bash
set -eo pipefail
: "${INTERNAL_TEACHER:?Set standalone teacher checkpoint}"
: "${INTERNAL_LABEL:?Set unique experiment label}"
AUDIT_ROOT=/scratch/wzhao20/opd-gate-audit-run-20260909
REPO=/scratch/wzhao20/DOGe-official
ROOT=/scratch/wzhao20/AKDA2/gsm_vocab_aligned_minillm_20260909
STUDENT=${INTERNAL_STUDENT:-$REPO/outputs/qwen2_5_0p5b_instruct_sft_14b_cot_gsm1000_correctonly_20260908/sft_training/final}
PROMPTS=/scratch/wzhao20/AKDA2/opd_antidistill_minillm/experiments/qwen2_5_0p5b_7b_sft_then_14b_opd_gsm_chatprompt_20260908/prompts
SAVE=$AUDIT_ROOT/results/opd_corrected_20260911/${INTERNAL_LABEL}_opd
[[ ! -e "$SAVE" ]] || { echo "Refusing overwrite $SAVE"; exit 2; }
source /scratch/wzhao20/AKDA2/opd_antidistill_minillm/env_scratch.sh
export PATH=/scratch/wzhao20/conda_envs/minillm_official/bin:$PATH
export PYTHONPATH=$ROOT:$REPO/src
export HF_HOME=/scratch/wzhao20/hf_cache
export HF_DATASETS_CACHE=$REPO/data/gsm8k_hf_cache_20260908
export OMP_NUM_THREADS=4
export MINILLM_TOKENIZER_PATH=$STUDENT
export MINILLM_PREFIX_HACK_MODE=none
export AUDIT_ARM=clean
export AUDIT_GATE_ALL_SIGNALS=0
export AUDIT_REPAIR_RUNTIME=1
export AUDIT_REPAIR_INPUT_MASK=1
export AUDIT_SKIP_ALL_INTERNAL_EVAL=1
export AUDIT_PAIR_TAG=corrected_20260911
export AUDIT_MINILLM_ROOT=$ROOT
# No likelihood gate is installed; no reference student is loaded by the teacher.
unset AUDIT_REFERENCE AUDIT_POISON AUDIT_VARIANT
cd "$ROOT"
torchrun --nproc_per_node 1 --master_addr localhost --master_port ${INTERNAL_PORT:-29977} "$AUDIT_ROOT/experiments/opd_corrected_20260911/simple_entry.py" \
  --base-path "$ROOT" --model-path "$STUDENT" --teacher-model-path "$INTERNAL_TEACHER" \
  --ckpt-name 0.5B-instruct-14b-sft --teacher-ckpt-name internalized-7B \
  --n-gpu 1 --n-nodes 1 --model-type qwen2 --teacher-model-fp16 --gradient-checkpointing \
  --prompt-data-dir "$PROMPTS" --dev-num 200 --num-workers 0 \
  --epochs 10 --total-iters ${BASELINE_STEPS:-240} --kd-ratio 0.5 \
  --batch-size 2 --lr ${BASELINE_LR:-1e-6} --lr-min ${BASELINE_LR:-1e-6} --gradient-accumulation-steps 2 \
  --max-length 640 --max-prompt-length 256 --warmup-iters 2 --scheduler-name constant_trm \
  --save "$SAVE" --seed ${INTERNAL_SEED:-10} --seed-ppo ${INTERNAL_PPO_SEED:-42} --seed-lm ${INTERNAL_LM_SEED:-7} \
  --save-interval ${CORRECTED_SAVE_INTERVAL:-120} --eval-interval 1000 --log-interval 5 --mid-log-num 1 \
  --type minillm --ppo-epochs 1 --num-rollouts 8 --chunk-size 2 --length-norm --single-step-reg \
  --reward-scaling 0.5 --cliprange-reward 100 --do-sample --top-k 0 --top-p 1.0 --temperature 1.0 \
  --deepspeed --deepspeed_config "$ROOT/configs/deepspeed/ds_config_zero1_bf16.json"
