#!/usr/bin/env bash
# 复现 Zhan et al. 2026 的 undistillable 校准: CSQA 上训 teacher(pi_C=未 SFT 0.5B 主学生)
# -> 并行: (a) 同学生 CSQA OPD 臂(80 步, 复用 mcq_csqa_05 的 initial/clean)  (b) teacher 保持: GSM200 / CSQA200 greedy+采样 + 输出审计
# 用法: JOB=<alloc> NAME=<tag> [STEPS=120] [MAXMIN=70] [SKIP_TRAIN=<model_dir>] bash run_calibknob.sh
set -uo pipefail
: "${JOB:?}"; : "${NAME:?}"
source /scratch/wzhao20/anti_opd_claude_20260914/experiments/claude_20260914/env.sh
E=$W/experiments/claude_20260914; K=$W/experiments/calibknob_20261011; C=$W/results/claude_20260914
RUN=$C/ck_${NAME}_$(date +%m%d_%H%M%S); mkdir -p $RUN
exec > >(tee -a $RUN/chain.log) 2>&1
SR="srun --jobid=$JOB --overlap --exact --cpu-bind=none -N1 -n1 -c4 --gres=gpu:1"
ORIG=/scratch/wzhao20/DOGe-official/models/qwen2.5-7b-instruct
PROXY=/scratch/wzhao20/DOGe-official/models/qwen2.5-0.5b-instruct
SLICE=$SRC/results/opd_update_20260911/static_entropy_slice600_audit_9915409_sft/gsm8k-results.json
echo "[$(date +%T)] calibknob $NAME job=$JOB steps=${STEPS:-120} maxmin=${MAXMIN:-70}"
if [ -n "${SKIP_TRAIN:-}" ]; then T=$SKIP_TRAIN; else
  $SR $PY $K/calib_grpo.py --teacher $ORIG --proxy $PROXY --output $RUN/teacher --mcq-dir $E \
      --steps ${STEPS:-120} --max-minutes ${MAXMIN:-70} > $RUN/teacher.log 2>&1 < /dev/null
  T=$RUN/teacher/model
fi
[ -f $T/config.json ] || { echo "teacher 训练失败"; tail -5 $RUN/teacher.log; exit 1; }
echo "[$(date +%T)] teacher=$T"; tail -2 $RUN/teacher/train_log.jsonl 2>/dev/null | cut -c1-300
# (a) 学生臂: 同一 0.5B(=pi_C) 在 CSQA 上 OPD
( STEPS=${ARM_STEPS:-80} JOB=$JOB TEACHER=$T NAME=ck_$NAME DATASET=csqa RUN=$RUN bash $E/mcq_opd_arm.sh > $RUN/arm.log 2>&1 ) &
# (b) teacher 保持: GSM(未见域) + CSQA(训练域)
$SR $PY $W/experiments/internalize_20260910/evaluate_plain.py --model $T --examples $SLICE --output $RUN/tgsm200 --limit 200 --modes greedy sampling --max-tokens 1024 > $RUN/tgsm200.log 2>&1 < /dev/null
cd $E && $PY $E/audit_outputs.py --dir $RUN/tgsm200 --kind teacher --cap 1024 --expect 200 --log $RUN/tgsm200.log --label CK_${NAME}_gsm200 | grep "^AUDIT" | cut -c1-260
for MD in "" "--sample"; do n=greedy; [ -n "$MD" ] && n=sample
  $SR $PY $E/mcq_eval.py --dataset csqa --model $T --output $RUN/csqa_$n --start 0 --count 200 $MD > $RUN/csqa_$n.log 2>&1 < /dev/null
  cd $E && $PY $E/audit_outputs.py --dir $RUN/csqa_$n --kind mcq --cap 512 --expect 200 --log $RUN/csqa_$n.log --label CK_${NAME}_csqa_$n | grep "^AUDIT" | cut -c1-260
done
echo "[$(date +%T)] teacher 评测完成(参照原 teacher: gsm200 91.5/91.5, csqa200 ~81/80)"
wait; echo "[$(date +%T)] 学生臂:"; grep -a "评测" $RUN/arm.log | tail -3 | cut -c1-200
touch $RUN/CK_DONE; echo "[$(date +%T)] 全部完成"
