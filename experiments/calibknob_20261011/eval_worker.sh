#!/usr/bin/env bash
# 评测卡调度: 跟随一条校准训练(RUN=ck_long_xxx), 优先评最新的、未评的 50 步整数倍 ckpt;
# 空闲超过 20 min 则回补最新的未评 25 步整数倍; 卡剩余 <75 min 或训练结束且无可评 ckpt 时退出.
# 用法: JOB=<alloc> RUN=<ck_long 目录> TAGP=<b04|b004> [REF=1] bash eval_worker.sh
set -uo pipefail
: "${JOB:?}"; : "${RUN:?}"; : "${TAGP:?}"
source /scratch/wzhao20/anti_opd_claude_20260914/experiments/claude_20260914/env.sh
K=$W/experiments/calibknob_20261011; E=$W/experiments/claude_20260914; C=$W/results/claude_20260914
EV=$C/ckeval_$TAGP; mkdir -p $EV/claims; exec >> $EV/worker.log 2>&1
SR="srun --jobid=$JOB --overlap --exact --cpu-bind=none -N1 -n1 -c4 --gres=gpu:1"
left () { squeue -h -j $JOB -o %L | awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else print $1 }'; }
echo "[$(date +%T)] worker job=$JOB run=$RUN left=$(left)"
if [ "${REF:-0}" = 1 ] && [ ! -f $C/ckeval_ref/ref_orig/EVAL_DONE ]; then   # 参照: 原 teacher 同口径 + 未训学生 GSM300
  mkdir -p $C/ckeval_ref
  ( [ -f $C/ckeval_ref/raw_student_gsm300/gsm8k-results.json ] || $SR $PY $W/experiments/baseline_20260911/evaluate.py --model /scratch/wzhao20/DOGe-official/models/qwen2.5-0.5b-instruct \
      --output $C/ckeval_ref/raw_student_gsm300 --split test --start 0 --count 300 --max-tokens 1024 > $C/ckeval_ref/raw_student_gsm300.log 2>&1 < /dev/null ) &
  JOB=$JOB D=$C/ckeval_ref/ref_orig TAG=ck_ref_orig MODEL=/scratch/wzhao20/DOGe-official/models/qwen2.5-7b-instruct NO_STU=1 bash $K/eval_ckpt.sh
  wait; echo "[$(date +%T)] 参照完成"
fi
pick () {  # $1 = 步数模数; 输出最新的未认领、已写完(>60s)的 ckpt 步数
  for d in $(ls -d $RUN/ckpt/step_* 2>/dev/null | awk -F_ '{print $NF" "$0}' | sort -k1,1nr | cut -d" " -f2); do
    s=${d##*_}; [ $((s % $1)) = 0 ] || continue; [ -f $d/adapter_model.safetensors ] || continue
    [ $(( $(date +%s) - $(stat -c %Y $d/adapter_model.safetensors) )) -gt 60 ] || continue
    [ -d $EV/claims/s$s ] && continue; echo $s; return
  done; }
idle=$(date +%s)
while true; do
  L=$(left); [ -z "$L" ] || [ "$L" -lt 75 ] && { echo "[$(date +%T)] 卡剩余 ${L:-0} min, 退出"; break; }
  S=$(pick 50); [ -z "$S" ] && [ $(( $(date +%s) - idle )) -gt 1200 ] && S=$(pick 25)
  if [ -z "$S" ]; then
    grep -a -q -E "time cap|DONE \(no merge\)" $RUN.train.log 2>/dev/null && [ -z "$(pick 25)" ] && { echo "[$(date +%T)] 训练已结束且无可评 ckpt, 退出"; break; }
    sleep 60; continue
  fi
  mkdir $EV/claims/s$S 2>/dev/null || continue
  echo "[$(date +%T)] 评 step $S (空闲 $(( ($(date +%s) - idle) / 60 )) min)"
  JOB=$JOB D=$EV/s$S TAG=ck_${TAGP}_s$S ADAPTER=$RUN/ckpt/step_$S bash $K/eval_ckpt.sh
  grep -a "^CKRES" $EV/s$S/eval.log | tail -1
  $PY $K/summarize_ck.py --all $C/ckeval_b04 $C/ckeval_b004 > $C/ckeval_table.txt 2>&1
  idle=$(date +%s)
done
