#!/usr/bin/env bash
# 无人值守第二阶段编排(用户离线): 主实验(b04/b004 训满或到时)结束后, 用剩余卡时做
#   ① 跨学生 Coder-0.5B 两臂(GSM/CSQA)  ② 新 β=0.2 训练 + 跟评.
# 卡 A=b04 训练卡, B=b004 训练卡, CA=b04 评测卡, CB=b004 评测卡. 用法: A= B= CA= CB= bash phase2.sh
set -uo pipefail
: "${A:?}"; : "${B:?}"; : "${CA:?}"; : "${CB:?}"
source /scratch/wzhao20/anti_opd_claude_20260914/experiments/claude_20260914/env.sh
K=$W/experiments/calibknob_20261011; C=$W/results/claude_20260914; LOG=$C/phase2.log; mkdir -p $C/ckeval_coder
log () { echo "[$(date '+%m-%d %T')] $*" >> $LOG; }
left () { squeue -h -j $1 -o %L | awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else print $1 }'; }
lft () { local l=$(left $1); echo ${l:-0}; }
done_train () { tail -5 $C/ck_long_$1.train.log 2>/dev/null | grep -a -q -E "time cap|DONE \(no merge\)"; }
worker_done () { tail -2 $C/ckeval_$1/worker.log 2>/dev/null | grep -a -q "退出"; }
last_ckpt () { ls -d $C/ck_long_$1/ckpt/step_* 2>/dev/null | awk -F_ '{print $NF}' | sort -n | tail -1; }
near_ckpt () { ls -d $C/ck_long_$1/ckpt/step_* 2>/dev/null | awk -F_ '{print $NF}' | sort -n | awk -v t=$2 '$1<=t' | tail -1; }   # <= 目标步的最近 ckpt
coder () {  # $1=卡 $2=run 名 $3=步
  local S=$3; [ -n "$S" ] || return; local D=$C/ckeval_coder/$2_s$S
  [ -f $D/CODER_DONE ] && return; mkdir -p $D/.claim 2>/dev/null; mkdir $D/.claim/x 2>/dev/null || return
  log "coder arms $2 s$S on $1 (left $(lft $1))"
  JOB=$1 D=$D TAG=$2_s$S ADAPTER=$C/ck_long_$2/ckpt/step_$S bash $K/coder_arms.sh
  log "coder done $2 s$S: $(grep -a CODERRES $D/coder.log | tail -1 | cut -c1-200)"; }
log "phase2 start A=$A B=$B CA=$CA CB=$CB"

( # 卡 A: 等 b04 训练结束 -> Coder(b04 末 ckpt) -> β=0.2 训练(看门狗续训)
  until done_train b04 || [ $(lft $A) -lt 60 ]; do sleep 300; done; log "A: b04 训练结束 (left $(lft $A))"
  sleep 120
  [ $(lft $A) -gt 110 ] && coder $A b04 $(last_ckpt b04)
  if [ $(lft $A) -gt 90 ]; then
    log "A: 启动 β=0.2 训练"; touch $C/ck_long_b02.STARTED
    JOB=$A OUT=$C/ck_long_b02 BETA=0.2 bash $K/train_long.sh
    JOB=$A OUT=$C/ck_long_b02 BETA=0.2 bash $K/train_watchdog.sh
  fi; log "A: 结束" ) &

( # 卡 CA: b04 跟评结束 -> Coder 参照(未训 Coder GSM300) + Coder(b04 中段 ~600) -> 跟评 β=0.2
  until worker_done b04 || [ $(lft $CA) -lt 60 ]; do sleep 300; done; log "CA: b04 跟评结束 (left $(lft $CA))"
  if [ ! -f $C/ckeval_ref/coder_gsm300/gsm8k-results.json ] && [ $(lft $CA) -gt 30 ]; then
    srun --jobid=$CA --overlap --exact --cpu-bind=none -N1 -n1 -c4 --gres=gpu:1 $PY $W/experiments/baseline_20260911/evaluate.py \
      --model $SRC/results/generalization_20260913/qwen2.5-coder-0.5b-instruct --output $C/ckeval_ref/coder_gsm300 --split test --start 0 --count 300 --max-tokens 1024 \
      > $C/ckeval_ref/coder_gsm300.log 2>&1 < /dev/null; log "CA: Coder GSM300 initial 参照完成"; fi
  [ $(lft $CA) -gt 110 ] && coder $CA b04 $(near_ckpt b04 600)
  until [ -f $C/ck_long_b02.STARTED ] || [ $(lft $CA) -lt 75 ]; do sleep 300; done
  [ $(lft $CA) -gt 75 ] && { log "CA: 跟评 β=0.2"; JOB=$CA RUN=$C/ck_long_b02 TAGP=b02 bash $K/eval_worker4.sh; }
  log "CA: 结束" ) &

( # 卡 B: 等 b004 训练结束 -> Coder(b004 末 ckpt)
  until done_train b004 || [ $(lft $B) -lt 60 ]; do sleep 300; done; log "B: b004 训练结束 (left $(lft $B))"
  sleep 120; [ $(lft $B) -gt 110 ] && coder $B b004 $(last_ckpt b004)
  log "B: 结束" ) &

( # 卡 CB: b004 跟评结束 -> Coder(b004 中段 ~600) -> Coder(两条 ~300)  [β=0.2 只由 CA 跟评, 避免认领冲突]
  until worker_done b004 || [ $(lft $CB) -lt 60 ]; do sleep 300; done; log "CB: b004 跟评结束 (left $(lft $CB))"
  [ $(lft $CB) -gt 110 ] && coder $CB b004 $(near_ckpt b004 600)
  [ $(lft $CB) -gt 110 ] && coder $CB b004 $(near_ckpt b004 300)
  [ $(lft $CB) -gt 110 ] && coder $CB b04 $(near_ckpt b04 300)
  log "CB: 结束" ) &
wait; log "phase2 全部结束"
