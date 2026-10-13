#!/usr/bin/env bash
# 补评 gsm_long 某条 run 已存在但未评测的学生存档: JOB=<alloc> NAME=<gsmlong 名> STEP=<步> bash fill_eval.sh
set -uo pipefail
: "${JOB:?}"; : "${NAME:?}"; : "${STEP:?}"
source /scratch/wzhao20/anti_opd_claude_20260914/experiments/claude_20260914/env.sh
E=$W/experiments/claude_20260914; K=$W/experiments/calibknob_20261011; C=$W/results/claude_20260914; D=$C/gsmlong/$NAME
SR="srun --jobid=$JOB --overlap --exact --cpu-bind=none -N1 -n1 -c4 --gres=gpu:1"
B=$(find $(cat $C/fine_gl_$NAME/opd_root) -maxdepth 2 -mindepth 2 -type d | head -1)
[ -f $B/$STEP/pytorch_model.bin ] || { echo "无存档 $NAME $STEP" >> $D/run.log; exit 1; }
[ -f $D/s$STEP/gsm8k-results.json ] && exit 0
rm -rf $D/s$STEP; echo "[$(date +%T)] fill eval $NAME s$STEP on $JOB" >> $D/run.log
$SR $PY $W/experiments/baseline_20260911/evaluate.py --model $B/$STEP --output $D/s$STEP --split test --start 0 --count 300 --max-tokens 1024 > $D/s$STEP.log 2>&1 < /dev/null
for i in $(seq 1 18); do [ -f $D/s$STEP/gsm8k-results.json ] && break; sleep 10; done
cd $E && $PY $E/audit_outputs.py --dir $D/s$STEP --kind gsm --cap 1024 --expect 300 --log $D/s$STEP.log --label gl_${NAME}_s$STEP | grep "^AUDIT" >> $D/run.log
$PY $K/gsmlong_summary.py > $C/gsmlong/table.txt 2>&1
