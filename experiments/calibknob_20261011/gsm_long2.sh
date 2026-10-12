#!/usr/bin/env bash
# 跨域 GSM 学生长程曲线(对齐原文 Figure 3 的 200 步): 未 SFT 0.5B, OPD STEPS 步(lr 1e-6), 每 SAVE 步存档,
# 存档一出现就在同卡评 test[0:300](1024 token)+审计. 结果 results/claude_20260914/gsmlong/<NAME>/s<step>/
# v2: 合并在 GPU 上进行(避免 32G 主机内存 OOM)
# 用法: JOB=<alloc> NAME=<名> (ADAPTER=<ckpt/step_N> | MODEL=<完整模型>) [STEPS=200] [SAVE=20] bash gsm_long.sh
set -uo pipefail
: "${JOB:?}"; : "${NAME:?}"
source /scratch/wzhao20/anti_opd_claude_20260914/experiments/claude_20260914/env.sh
E=$W/experiments/claude_20260914; K=$W/experiments/calibknob_20261011; C=$W/results/claude_20260914
SR="srun --jobid=$JOB --overlap --exact --cpu-bind=none -N1 -n1 -c4 --gres=gpu:1"
ORIG=/scratch/wzhao20/DOGe-official/models/qwen2.5-7b-instruct; RAW=/scratch/wzhao20/DOGe-official/models/qwen2.5-0.5b-instruct
STEPS=${STEPS:-200}; SAVE=${SAVE:-20}; D=$C/gsmlong/$NAME; mkdir -p $D; exec >> $D/run.log 2>&1
echo "[$(date +%T)] gsm_long $NAME job=$JOB steps=$STEPS save=$SAVE adapter=${ADAPTER:-} model=${MODEL:-}"
if [ -n "${ADAPTER:-}" ]; then
  T=$D/teacher; [ -f $T/config.json ] || $SR $PY $K/merge_ckpt.py --base $ORIG --adapter $ADAPTER --output $T --device cuda < /dev/null
else T=$MODEL; fi
for i in $(seq 1 30); do [ -f $T/config.json ] && break; sleep 10; done
[ -f $T/config.json ] || { echo "无模型 $T"; exit 1; }
( JOB=$JOB TEACHER=$T NAME=gl_$NAME STEPS=$STEPS SAVE=$SAVE STU=$RAW LR=1e-6 bash $E/fine_sweep_train2.sh > $D/train.log 2>&1 < /dev/null ) &
TP=$!
evalone () {  # $1 = step
  local s=$1; [ -f $D/s$s/gsm8k-results.json ] && return 0
  local B=$(find $(cat $C/fine_gl_$NAME/opd_root 2>/dev/null) -maxdepth 2 -mindepth 2 -type d 2>/dev/null | head -1)
  [ -n "$B" ] && [ -f $B/$s/pytorch_model.bin ] || return 1
  [ $(( $(date +%s) - $(stat -c %Y $B/$s/pytorch_model.bin) )) -gt 60 ] || return 1
  $SR $PY $W/experiments/baseline_20260911/evaluate.py --model $B/$s --output $D/s$s --split test --start 0 --count 300 --max-tokens 1024 > $D/s$s.log 2>&1 < /dev/null
  for i in $(seq 1 18); do [ -f $D/s$s/gsm8k-results.json ] && break; sleep 10; done
  cd $E && $PY $E/audit_outputs.py --dir $D/s$s --kind gsm --cap 1024 --expect 300 --log $D/s$s.log --label gl_${NAME}_s$s | grep "^AUDIT"
}
while true; do
  for s in $(seq $SAVE $SAVE $STEPS); do evalone $s; done
  left=0; for s in $(seq $SAVE $SAVE $STEPS); do [ -f $D/s$s/gsm8k-results.json ] || left=1; done
  [ $left = 0 ] && break
  kill -0 $TP 2>/dev/null || { for s in $(seq $SAVE $SAVE $STEPS); do evalone $s; done; break; }
  sleep 60
done
wait $TP
[ -n "${ADAPTER:-}" ] && rm -rf $D/teacher
$PY $K/gsmlong_summary.py > $C/gsmlong/table.txt 2>&1
touch $D/DONE; echo "[$(date +%T)] done $NAME"
