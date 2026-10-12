#!/usr/bin/env bash
# 长程校准(可续训): 只训练 + 每 SAVE 步存 adapter. 时间上限按卡剩余时间自动设(留 15 min 余量).
# 用法: JOB=<alloc> OUT=<dir> [BETA=0.04] [STEPS=1218] [SAVE=25] [RESUME=1] bash train_long.sh
set -uo pipefail
: "${JOB:?}"; : "${OUT:?}"
source /scratch/wzhao20/anti_opd_claude_20260914/experiments/claude_20260914/env.sh
K=$W/experiments/calibknob_20261011; E=$W/experiments/claude_20260914
SR="srun --jobid=$JOB --overlap --exact --cpu-bind=none -N1 -n1 -c4 --gres=gpu:1"
LEFT=$(squeue -h -j $JOB -o %L)   # [D-]HH:MM:SS
MIN=$(echo "$LEFT" | awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else print $1 }')
MAXMIN=$(( MIN - 15 )); mkdir -p $(dirname $OUT)
echo "[$(date +%T)] train_long job=$JOB left=$LEFT maxmin=$MAXMIN beta=${BETA:-0.04} resume=${RESUME:-0}" | tee -a $OUT/../$(basename $OUT).chain.log
$SR $PY $K/calib_grpo.py --teacher /scratch/wzhao20/DOGe-official/models/qwen2.5-7b-instruct \
    --proxy /scratch/wzhao20/DOGe-official/models/qwen2.5-0.5b-instruct --mcq-dir $E --output $OUT \
    --steps ${STEPS:-1218} --max-minutes $MAXMIN --beta ${BETA:-0.04} --save-every ${SAVE:-25} --no-merge \
    $( [ "${RESUME:-0}" = 1 ] && echo --resume ) >> $OUT.train.log 2>&1 < /dev/null
echo "[$(date +%T)] 退出: $(grep -a -E 'CKPT|time cap|DONE|Traceback|EPOCH' $OUT.train.log | tail -3 | tr '\n' ' ')" | tee -a $OUT/../$(basename $OUT).chain.log
