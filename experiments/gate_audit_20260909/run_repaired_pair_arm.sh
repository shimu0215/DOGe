#!/usr/bin/env bash
set -eo pipefail
 : "${AUDIT_VARIANT:?Set clean or contrast}"
case "$AUDIT_VARIANT" in
  clean) export AUDIT_ARM=legacy ;;
  contrast) export AUDIT_ARM=likelihood ;;
  *) exit 2 ;;
esac
export AUDIT_POISON=contrast
export AUDIT_CONTRAST_BETA=4
export AUDIT_SKIP_ALL_INTERNAL_EVAL=1
export AUDIT_PAIR_TAG=repaired_pair_20260909
export AUDIT_REPAIR_RUNTIME=1
export AUDIT_REPAIR_INPUT_MASK=1
AUDIT_LABEL=repaired_${AUDIT_VARIANT}
case "$AUDIT_ARM" in legacy|unified|likelihood) ;; *) exit 2;; esac
AUDIT_ROOT=/scratch/wzhao20/opd-gate-audit-run-20260909
REPO=/scratch/wzhao20/DOGe-official
ROOT=/scratch/wzhao20/AKDA2/gsm_vocab_aligned_minillm_20260909
STUDENT=$REPO/outputs/qwen2_5_0p5b_instruct_sft_14b_cot_gsm1000_correctonly_20260908/sft_training/final
TEACHER=$REPO/models/qwen2.5-7b-instruct
PROMPTS=/scratch/wzhao20/AKDA2/opd_antidistill_minillm/experiments/qwen2_5_0p5b_7b_sft_then_14b_opd_gsm_chatprompt_20260908/prompts
SAVE=$AUDIT_ROOT/results/${AUDIT_LABEL}_t120
[[ ! -e "$SAVE" ]] || { echo "Refusing to overwrite $SAVE"; exit 2; }
source /scratch/wzhao20/AKDA2/opd_antidistill_minillm/env_scratch.sh
export PATH=/scratch/wzhao20/conda_envs/minillm_official/bin:$PATH
export PYTHONPATH=$ROOT:$REPO/src
export HF_HOME=/scratch/wzhao20/hf_cache
export HF_DATASETS_CACHE=$REPO/data/gsm8k_hf_cache_20260908
export OMP_NUM_THREADS=4
export MINILLM_TOKENIZER_PATH=$STUDENT
export MINILLM_PREFIX_HACK_MODE=imposs_gate
export MINILLM_IMPOSS_TAU=${AUDIT_TAU:-0.9287033677}
export MINILLM_IMPOSS_WIN=${AUDIT_WIN:-8}
export MINILLM_IMPOSS_SHARP=${AUDIT_SHARP:-0.5}
export MINILLM_IMPOSS_MARGIN=${AUDIT_MARGIN:-0}
export MINILLM_IMPOSS_DECOY_RANK=1
export MINILLM_IMPOSS_PROTECT_EOS=1
export MINILLM_IMPOSS_LOWENT=0
export MINILLM_IMPOSS_SIGNAL=decoy
if [[ "$AUDIT_VARIANT" == clean ]]; then export MINILLM_PREFIX_HACK_MODE=none; fi
export AUDIT_MINILLM_ROOT=$ROOT
export AUDIT_GATE_ALL_SIGNALS=0
[[ "$AUDIT_ARM" == legacy ]] || export AUDIT_GATE_ALL_SIGNALS=1
export AUDIT_REFERENCE=$STUDENT
cd "$ROOT"
torchrun --nproc_per_node 1 --master_addr localhost --master_port 29961 "$AUDIT_ROOT/experiments/gate_audit_20260909/train_entry.py" \
  --base-path "$ROOT" --model-path "$STUDENT" --teacher-model-path "$TEACHER" \
  --ckpt-name 0.5B-instruct-14b-sft --teacher-ckpt-name 7B-instruct \
  --n-gpu 1 --n-nodes 1 --model-type qwen2 --teacher-model-fp16 --gradient-checkpointing \
  --prompt-data-dir "$PROMPTS" --dev-num 200 --num-workers 0 \
  --epochs 10 --total-iters 120 --kd-ratio 0.5 \
  --batch-size 2 --lr 5e-7 --lr-min 5e-7 --gradient-accumulation-steps 2 \
  --max-length 640 --max-prompt-length 256 --warmup-iters 2 --scheduler-name constant_trm \
  --save "$SAVE" --seed 10 --seed-ppo 42 --seed-lm 7 \
  --save-interval 40 --eval-interval 1000 --log-interval 5 --mid-log-num 1 \
  --type minillm --ppo-epochs 1 --num-rollouts 8 --chunk-size 2 --length-norm --single-step-reg \
  --reward-scaling 0.5 --cliprange-reward 100 --do-sample --top-k 0 --top-p 1.0 --temperature 1.0 \
  --deepspeed --deepspeed_config "$ROOT/configs/deepspeed/ds_config_zero1_bf16.json"
FINAL=$(find "$SAVE" -type f -path '*/120/pytorch_model.bin' -printf '%h\n')
[[ -n "$FINAL" && "$FINAL" != *$'\n'* ]]
python "$REPO/scripts/generate-eval-qwen2_5.py" --model_path "$FINAL" \
  --output_dir "$AUDIT_ROOT/results/${AUDIT_LABEL}_gsm200" --limit 200 --batch_size 8 --max_tokens 512
