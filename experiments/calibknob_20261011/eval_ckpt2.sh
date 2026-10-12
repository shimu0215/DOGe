#!/usr/bin/env bash
# 评一个 teacher(合并好的模型或 adapter ckpt), 两路并行:
#  路1: teacher 小规模评测(GSM100 greedy/raw, CSQA100 greedy/raw) -> 同域学生 CSQA OPD 80 步 + 300 题
#  路2: 跨域学生 GSM OPD 60 步 + GSM test[0:300]
# v2: 合并后等待 config.json 可见(NFS 延迟)
# 用法: JOB=<alloc> D=<输出目录> TAG=<唯一名> (ADAPTER=<ckpt/step_N> | MODEL=<完整模型>) bash eval_ckpt2.sh
set -uo pipefail
: "${JOB:?}"; : "${D:?}"; : "${TAG:?}"
source /scratch/wzhao20/anti_opd_claude_20260914/experiments/claude_20260914/env.sh
E=$W/experiments/claude_20260914; K=$W/experiments/calibknob_20261011; C=$W/results/claude_20260914
SR="srun --jobid=$JOB --overlap --exact --cpu-bind=none -N1 -n1 -c4 --gres=gpu:1"
ORIG=/scratch/wzhao20/DOGe-official/models/qwen2.5-7b-instruct
RAW=/scratch/wzhao20/DOGe-official/models/qwen2.5-0.5b-instruct
SLICE=$SRC/results/opd_update_20260911/static_entropy_slice600_audit_9915409_sft/gsm8k-results.json
mkdir -p $D; exec >> $D/eval.log 2>&1
echo "[$(date +%T)] eval $TAG job=$JOB adapter=${ADAPTER:-} model=${MODEL:-}"
if [ -n "${ADAPTER:-}" ]; then
  T=$D/model; [ -f $T/config.json ] || $SR $PY $K/merge_ckpt.py --base $ORIG --adapter $ADAPTER --output $T < /dev/null
else T=$MODEL; fi
for i in $(seq 1 30); do [ -f $T/config.json ] && break; sleep 10; done   # NFS 落盘延迟
[ -f $T/config.json ] || { echo "无模型 $T"; exit 1; }
gsm_arm () {
  [ -f $D/gsm_stu/gsm8k-results.json ] && return
  JOB=$JOB TEACHER=$T NAME=$TAG STEPS=60 SAVE=60 STU=$RAW LR=1e-6 bash $E/fine_sweep_train2.sh > $D/gsm_arm_train.log 2>&1 < /dev/null
  M=$(find $(cat $C/fine_$TAG/opd_root) -path "*/60/pytorch_model.bin" 2>/dev/null | head -1 | xargs -r dirname)
  [ -n "$M" ] || { echo "GSM 臂无权重"; return; }
  $SR $PY $W/experiments/baseline_20260911/evaluate.py --model $M --output $D/gsm_stu --split test --start 0 --count 300 --max-tokens 1024 > $D/gsm_stu.log 2>&1 < /dev/null
  for i in $(seq 1 18); do [ -f $D/gsm_stu/gsm8k-results.json ] && break; sleep 10; done
  cd $E && $PY $E/audit_outputs.py --dir $D/gsm_stu --kind gsm --cap 1024 --expect 300 --log $D/gsm_stu.log --label ${TAG}_gsm_stu | grep "^AUDIT"
}
teacher_then_csqa () {
  [ -d $D/tgsm ] || $SR $PY $W/experiments/internalize_20260910/evaluate_plain.py --model $T --examples $SLICE --output $D/tgsm --limit 100 --modes greedy raw --max-tokens 1024 > $D/tgsm.log 2>&1 < /dev/null
  grep -a "^RESULT" $D/tgsm.log | cut -c1-160
  for m in greedy raw; do
    [ -f $D/tcsqa_$m/predictions.json ] || $SR $PY $K/mcq_eval_t.py --dataset csqa --model $T --output $D/tcsqa_$m --mode $m --count 100 > $D/tcsqa_$m.log 2>&1 < /dev/null
    tail -1 $D/tcsqa_$m.log
  done
  [ "${NO_STU:-0}" = 1 ] && return
  STEPS=80 JOB=$JOB TEACHER=$T NAME=$TAG DATASET=csqa RUN=$D bash $E/mcq_opd_arm.sh > $D/csqa_arm.log 2>&1 < /dev/null
  P=$C/mcq_csqa_05/$TAG/predictions.json; [ -f $P ] && cd $E && $PY $E/audit_outputs.py --dir $C/mcq_csqa_05/$TAG --kind mcq --cap 512 --expect 300 --label ${TAG}_csqa_stu | grep "^AUDIT"
}
teacher_then_csqa & P1=$!
[ "${NO_STU:-0}" = 1 ] || { gsm_arm & P2=$!; }
wait
$PY $K/summarize_ck.py --one $D --tag $TAG
[ -n "${ADAPTER:-}" ] && [ "${KEEP_MODEL:-0}" != 1 ] && rm -rf $D/model
touch $D/EVAL_DONE; echo "[$(date +%T)] done $TAG"
