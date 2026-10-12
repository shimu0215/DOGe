#!/usr/bin/env bash
# 训练看门狗: 每 5 min 查一次; 若该卡上没有 calib_grpo 进程、训练日志未正常结束(time cap / DONE)、卡剩余 >40 min, 则 RESUME=1 续训.
# 同一 OUT 连续自动续训 >3 次则停手(避免崩溃循环). 用法: JOB=<alloc> OUT=<ck_long 目录> BETA=<β> bash train_watchdog.sh
set -uo pipefail
: "${JOB:?}"; : "${OUT:?}"; : "${BETA:?}"
source /scratch/wzhao20/anti_opd_claude_20260914/experiments/claude_20260914/env.sh
K=$W/experiments/calibknob_20261011; LOG=$OUT.watchdog.log; n=0
left () { squeue -h -j $JOB -o %L | awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else print $1 }'; }
echo "[$(date +%T)] watchdog job=$JOB out=$OUT beta=$BETA" >> $LOG
while true; do
  sleep 300
  L=$(left); [ -z "$L" ] || [ "$L" -lt 40 ] && { echo "[$(date +%T)] 卡剩余 ${L:-0} min, 退出" >> $LOG; break; }
  tail -5 $OUT.train.log | grep -a -q -E "time cap|DONE \(no merge\)" && { echo "[$(date +%T)] 训练正常结束, 退出" >> $LOG; break; }
  P=$(srun --jobid=$JOB --overlap -N1 -n1 pgrep -f "[c]alib_grpo.py" 2>/dev/null | head -1)
  [ -n "$P" ] && continue
  sleep 60; P=$(srun --jobid=$JOB --overlap -N1 -n1 pgrep -f "[c]alib_grpo.py" 2>/dev/null | head -1); [ -n "$P" ] && continue
  n=$((n+1)); [ $n -gt 3 ] && { echo "[$(date +%T)] 已自动续训 3 次仍崩, 停手" >> $LOG; break; }
  echo "[$(date +%T)] 训练进程不在且未正常结束 -> 第 $n 次续训(最近 ckpt: $(grep -a '^CKPT' $OUT.train.log | tail -1))" >> $LOG
  JOB=$JOB OUT=$OUT BETA=$BETA RESUME=1 bash $K/train_long.sh
done
