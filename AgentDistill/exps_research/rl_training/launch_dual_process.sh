#!/usr/bin/env bash
# =============================================================================
# launch_dual_process.sh — Dual-process-group OS-RL training launcher
#
# Architecture:
#   Process group A: accelerate (ZeRO-3, 4 ranks, GPU 0-3) — training
#   Process group B: resample_server.py (vLLM tp=4, GPU 0-3) — inference
#
# Both processes are launched from this shell script into the SAME SLURM
# allocation.  They run alternately (not simultaneously) coordinated via
# file signals in OUTPUT_DIR.
#
# Key reason this works vs. the legacy approach:
#   The resample server is launched from the shell (before training CUDA init),
#   so it has a CLEAN CUDA context on all 4 GPUs.  When it spawns a vLLM
#   subprocess it can initialise tp=4 without "invalid device ordinal" errors.
#
# Usage:
#   Adjust the variables below, then:
#     sbatch --jobid=<JOBID> launch_dual_process.sh
#   or run inside an already-running interactive job:
#     bash launch_dual_process.sh
# =============================================================================

set -euo pipefail

# ── User-configurable paths ──────────────────────────────────────────────────
MODEL_NAME="Qwen/Qwen3-32B"
OUTPUT_DIR="training_outputs/qwen3-32B/os_rl_full_v3"
TRAJECTORY_DIR="data/trajectories/os_rl_offline"
PILOT_QUESTIONS="data/pilot_questions/math500_pilot.json"
DEEPSPEED_CONFIG="exps_research/rl_training/ds_zero3_cpu_offload.json"
ACCEL_CONFIG="exps_research/rl_training/accel_config.yaml"

# ── Training hyper-parameters ────────────────────────────────────────────────
MAX_STEPS=500
RESAMPLE_EVERY=50
N_RESAMPLE_QUESTIONS=100
SEEDS_PER_RESAMPLE=2
MAX_AGENT_STEPS=5
TP_SIZE=4                   # tensor_parallel_size for resample server
CHECKPOINT_EVERY=100

# ── Environment ──────────────────────────────────────────────────────────────
export TORCH_NCCL_ENABLE_MONITORING=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# All 4 GPUs available to both process groups (they alternate, not concurrent)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ── Repo root (AgentDistill/) ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

echo "=== Dual-process-group OS-RL launcher ==="
echo "  OUTPUT_DIR  : ${OUTPUT_DIR}"
echo "  MODEL_NAME  : ${MODEL_NAME}"
echo "  TP_SIZE     : ${TP_SIZE}"
echo "  MAX_STEPS   : ${MAX_STEPS}"
echo "  RESAMPLE_EVERY: ${RESAMPLE_EVERY}"
date

mkdir -p "${OUTPUT_DIR}"

# ── Launch resample server (background, separate process group) ──────────────
# This process has a clean CUDA context — NOT forked from any training rank.
echo "[launcher] Starting resample server (n_gpus=${TP_SIZE}, parallel tp=1 per seed)..."
python -m exps_research.rl_training.resample_server \
    --work_dir "${OUTPUT_DIR}" \
    --n_gpus   "${TP_SIZE}"   \
    --poll_interval 5         \
    > "${OUTPUT_DIR}/resample_server.log" 2>&1 &
RESAMPLE_SERVER_PID=$!
echo "[launcher] Resample server PID: ${RESAMPLE_SERVER_PID}"

# Give the server a moment to start before training begins
sleep 3

# ── Launch training (foreground, accelerate ZeRO-3) ──────────────────────────
echo "[launcher] Starting training (accelerate ZeRO-3, 4 GPUs)..."
accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    -m exps_research.rl_training.train_os_rl_online_pilot \
        --trajectory_dir        "${TRAJECTORY_DIR}"       \
        --pilot_question_json   "${PILOT_QUESTIONS}"      \
        --model_name            "${MODEL_NAME}"            \
        --output_dir            "${OUTPUT_DIR}"            \
        --max_steps             "${MAX_STEPS}"             \
        --resample_every        "${RESAMPLE_EVERY}"        \
        --n_resample_questions  "${N_RESAMPLE_QUESTIONS}"  \
        --seeds_per_resample    "${SEEDS_PER_RESAMPLE}"    \
        --max_agent_steps       "${MAX_AGENT_STEPS}"       \
        --checkpoint_every      "${CHECKPOINT_EVERY}"      \
        --use_resample_server                              \
        --log_every 5

TRAIN_RC=$?
echo "[launcher] Training finished (rc=${TRAIN_RC})"
date

# ── Graceful shutdown: send shutdown signal to resample server ────────────────
echo "[launcher] Sending shutdown signal to resample server..."
SHUTDOWN_FILE="${OUTPUT_DIR}/resample_request.json"
echo '{"shutdown": true, "step": -1, "checkpoint_dir": "", "model_name": "", "seeds_and_questions": []}' \
    > "${SHUTDOWN_FILE}"

# Wait up to 30s for graceful exit, then kill
for i in $(seq 1 30); do
    if ! kill -0 "${RESAMPLE_SERVER_PID}" 2>/dev/null; then
        echo "[launcher] Resample server exited cleanly."
        break
    fi
    sleep 1
done
if kill -0 "${RESAMPLE_SERVER_PID}" 2>/dev/null; then
    echo "[launcher] Killing resample server (PID ${RESAMPLE_SERVER_PID})..."
    kill "${RESAMPLE_SERVER_PID}" 2>/dev/null || true
fi

echo "[launcher] All done."
exit "${TRAIN_RC}"
