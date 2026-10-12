#!/usr/bin/env bash
# 跨学生(未见学生 Qwen2.5-Coder-0.5B-Instruct)两臂, 并行:
#  GSM: fine_sweep_train_coder.sh 60 步(lr 3e-6) -> test[0:300]; 参照 fine_coder_clean/s60 前 300 题
#  CSQA: run_mcq_base_coder.sh 80 步(lr 3e-6) -> 300 题; 参照 mcq_csqa_coder initial 31.0 / clean 33.7
# 用法: JOB=<alloc> D=<输出目录> TAG=<唯一名> (ADAPTER=<ckpt/step_N> | MODEL=<完整模型>) bash coder_arms.sh
set -uo pipefail
: "${JOB:?}"; : "${D:?}"; : "${TAG:?}"
source /scratch/wzhao20/anti_opd_claude_20260914/experiments/claude_20260914/env.sh
E=$W/experiments/claude_20260914; K=$W/experiments/calibknob_20261011; C=$W/results/claude_20260914
SR="srun --jobid=$JOB --overlap --exact --cpu-bind=none -N1 -n1 -c4 --gres=gpu:1"
ORIG=/scratch/wzhao20/DOGe-official/models/qwen2.5-7b-instruct
CODER=$SRC/results/generalization_20260913/qwen2.5-coder-0.5b-instruct
mkdir -p $D; exec >> $D/coder.log 2>&1
echo "[$(date +%T)] coder arms $TAG job=$JOB adapter=${ADAPTER:-} model=${MODEL:-}"
if [ -n "${ADAPTER:-}" ]; then
  T=$D/model; [ -f $T/config.json ] || $SR $PY $K/merge_ckpt.py --base $ORIG --adapter $ADAPTER --output $T < /dev/null
else T=$MODEL; fi
for i in $(seq 1 30); do [ -f $T/config.json ] && break; sleep 10; done
[ -f $T/config.json ] || { echo "无模型 $T"; exit 1; }
gsm () {
  [ -f $D/coder_gsm/gsm8k-results.json ] && return
  JOB=$JOB TEACHER=$T NAME=ckc_$TAG STEPS=60 SAVE=60 bash $E/fine_sweep_train_coder.sh > $D/coder_gsm_train.log 2>&1 < /dev/null
  M=$(find $(cat $C/fine_ckc_$TAG/opd_root) -path "*/60/pytorch_model.bin" 2>/dev/null | head -1 | xargs -r dirname)
  [ -n "$M" ] || { echo "Coder GSM 臂无权重"; return; }
  $SR $PY $W/experiments/baseline_20260911/evaluate.py --model $M --output $D/coder_gsm --split test --start 0 --count 300 --max-tokens 1024 > $D/coder_gsm.log 2>&1 < /dev/null
  for i in $(seq 1 18); do [ -f $D/coder_gsm/gsm8k-results.json ] && break; sleep 10; done
  cd $E && $PY $E/audit_outputs.py --dir $D/coder_gsm --kind gsm --cap 1024 --expect 300 --log $D/coder_gsm.log --label ${TAG}_coder_gsm | grep "^AUDIT"
}
csqa () {
  [ -f $C/mcq_csqa_coder/ckc_$TAG/predictions.json ] && return
  STEPS=80 JOB=$JOB STUDENT=$CODER SN=coder DATASET=csqa ANTI="ckc_$TAG=$T" bash $E/run_mcq_base_coder.sh > $D/coder_csqa.log 2>&1 < /dev/null
  [ -f $C/mcq_csqa_coder/ckc_$TAG/predictions.json ] && cd $E && $PY $E/audit_outputs.py --dir $C/mcq_csqa_coder/ckc_$TAG --kind mcq --cap 512 --expect 300 --label ${TAG}_coder_csqa | grep "^AUDIT"
}
gsm & csqa & wait
$PY $K/summarize_ck.py --coder $D --tag $TAG
[ -n "${ADAPTER:-}" ] && rm -rf $D/model
touch $D/CODER_DONE; echo "[$(date +%T)] coder done $TAG"
