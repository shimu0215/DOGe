#!/bin/bash
# Watches for idle SLURM nodes and launches 14B sr_only experiments.
# Run on Hopper login node inside tmux:
#   tmux new -s watcher
#   bash AgentDistill/watcher_14b_sr_only.sh

AKDA=/scratch/wzhao20/AKDA2
STATE_DIR=$AKDA/watcher_state
LOG_DIR=$AKDA/logs
mkdir -p "$STATE_DIR" "$LOG_DIR"

# Ordered queue: each entry is "tag:c_value"
# Once launched, a tag file in STATE_DIR prevents re-launch.
EXPERIMENTS=(
    "c1p5:1.5"
    "c2p0:2.0"
    "c2p5:2.5"
    "c3p0:3.0"
)

log() { echo "[watcher $(date '+%H:%M:%S')] $*"; }

# Returns the next un-launched experiment tag:c, or empty string if all done.
next_experiment() {
    for entry in "${EXPERIMENTS[@]}"; do
        tag="${entry%%:*}"
        [ ! -f "$STATE_DIR/launched_14b_sr_only_$tag" ] && echo "$entry" && return
    done
}

# Returns 0 if the node has a running accelerate training process.
node_busy() {
    local node=$1
    ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" \
        "pgrep -u wzhao20 -f 'accelerate launch' > /dev/null 2>&1"
}

log "watcher started — monitoring for idle nodes"
log "experiments: ${EXPERIMENTS[*]}"

while true; do
    exp=$(next_experiment)
    if [ -z "$exp" ]; then
        log "all 4 experiments launched — watcher done"
        exit 0
    fi

    # Collect all unique nodes from running SLURM jobs
    running_nodes=$(squeue -u wzhao20 -h -t R -o '%N' 2>/dev/null | sort -u)

    for node in $running_nodes; do
        exp=$(next_experiment)
        [ -z "$exp" ] && break

        if node_busy "$node"; then
            log "$node is busy — skipping"
            continue
        fi

        tag="${exp%%:*}"
        c_val="${exp##*:}"
        script="$AKDA/AgentDistill/run_train_14b_sr_only_a0p5_b1_${tag}.sh"
        logfile="$LOG_DIR/14b_sr_only_${tag}.log"

        log "$node is idle — launching sr_only c=${c_val} (tag=${tag})"

        # Mark launched before SSH to avoid double-launch on watcher restart
        touch "$STATE_DIR/launched_14b_sr_only_$tag"

        ssh -o ConnectTimeout=10 -o BatchMode=yes "$node" \
            "nohup bash $script > $logfile 2>&1 < /dev/null &" < /dev/null
        ssh_exit=$?

        # exit 255 with Broken pipe is a false negative when using & — the
        # process is already running. Verify by checking if accelerate launched.
        if [ $ssh_exit -eq 0 ] || [ $ssh_exit -eq 255 ]; then
            sleep 5
            if ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" \
                   "pgrep -u wzhao20 -f 'accelerate launch' > /dev/null 2>&1" < /dev/null; then
                log "launched $tag on $node — log: $logfile"
            else
                log "ERROR: process not found on $node after launch attempt — removing state file"
                rm -f "$STATE_DIR/launched_14b_sr_only_$tag"
            fi
        else
            log "ERROR: ssh to $node failed (exit $ssh_exit) — removing state file"
            rm -f "$STATE_DIR/launched_14b_sr_only_$tag"
        fi

        sleep 10
    done

    log "sleeping 120s..."
    sleep 120
done
