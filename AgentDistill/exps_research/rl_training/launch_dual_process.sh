#!/usr/bin/env bash
# =============================================================================
# launch_dual_process.sh — Persistent-vLLM OS-RL training launcher
#
# Architecture:
#   Process A: resample_server.py  — starts a persistent vLLM server (tp=3,
#              GPU1/2/3) with --enable-lora.  Hot-swaps the LoRA adapter on
#              each resample cycle instead of reloading the full model.
#   Process B: accelerate (ZeRO-3 CPU-offload, 4 ranks, GPU0-3) — training
#
# Launch order:
#   1. resample_server.py starts vLLM, waits for /health → writes server_ready
#   2. This script polls for server_ready, then starts training
#   3. Training and vLLM coexist on GPU1/2/3 (ZeRO-3 CPU-offload keeps GPU
#      usage lean: ~3-5 GB during barrier, ~8 GB peak during forward/backward)
#
# Memory profile (GPU1/2/3 each, 81 GB A100):
#   vLLM model weights:   ~21 GB  (32B bf16 / 3 GPUs)
#   Training ZeRO-3 peak: ~8 GB
#   Total peak:           ~29 GB  → leaves ~52 GB for KV cache
#
# Usage:
#   Adjust variables below, then run inside an active SLURM allocation:
#     bash launch_dual_process.sh
# =============================================================================

set -euo pipefail

# ── User-configurable paths ──────────────────────────────────────────────────
MODEL_NAME="Qwen/Qwen3-32B"
OUTPUT_DIR="training_outputs/qwen3-32B/os_rl_full_v3"
TRAJECTORY_DIR="data/trajectories/os_rl_offline"
PILOT_QUESTIONS="data/pilot_questions/math500_pilot.json"
ACCEL_CONFIG="exps_research/mp_configs/accel_ds3_4gpu.yaml"

# ── Server configuration ─────────────────────────────────────────────────────
VLLM_GPU_IDS="1,2,3"               # physical GPUs for vLLM server
GPU_MEMORY_UTILIZATION="0.8"       # 0.8×81 GB ≈ 65 GB; leaves ~16 GB for training
MAX_LORA_RANK="64"                 # must be >= training --lora_r
MAX_MODEL_LEN="24576"

# ── Training hyper-parameters ────────────────────────────────────────────────
MAX_STEPS=500
RESAMPLE_EVERY=50
N_RESAMPLE_QUESTIONS=100
SEEDS_PER_RESAMPLE=2
MAX_AGENT_STEPS=5
CHECKPOINT_EVERY=100

# ── Environment ──────────────────────────────────────────────────────────────
export TORCH_NCCL_ENABLE_MONITORING=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ── Repo root (AgentDistill/) ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

echo "=== Persistent-vLLM OS-RL launcher ==="
echo "  OUTPUT_DIR              : ${OUTPUT_DIR}"
echo "  MODEL_NAME              : ${MODEL_NAME}"
echo "  VLLM_GPU_IDS            : ${VLLM_GPU_IDS}"
echo "  GPU_MEMORY_UTILIZATION  : ${GPU_MEMORY_UTILIZATION}"
echo "  MAX_STEPS               : ${MAX_STEPS}"
echo "  RESAMPLE_EVERY          : ${RESAMPLE_EVERY}"
date

mkdir -p "${OUTPUT_DIR}"

# Clean up stale signal files from any previous run
rm -f "${OUTPUT_DIR}/resample_request.json" \
      "${OUTPUT_DIR}/resample_done.json"    \
      "${OUTPUT_DIR}/server_ready"

# ── Launch resample server (background) ──────────────────────────────────────
# The server starts vLLM, waits for it to be healthy, then writes server_ready.
echo "[launcher] Starting resample server (vLLM tp=3, GPU${VLLM_GPU_IDS})..."
python -m exps_research.rl_training.resample_server \
    --work_dir                 "${OUTPUT_DIR}"               \
    --model_name               "${MODEL_NAME}"               \
    --gpu_ids                  "${VLLM_GPU_IDS}"             \
    --gpu_memory_utilization   "${GPU_MEMORY_UTILIZATION}"   \
    --max_lora_rank            "${MAX_LORA_RANK}"            \
    --max_model_len            "${MAX_MODEL_LEN}"            \
    --poll_interval            5                             \
    > "${OUTPUT_DIR}/resample_server.log" 2>&1 &
RESAMPLE_SERVER_PID=$!
echo "[launcher] Resample server PID: ${RESAMPLE_SERVER_PID}"

# ── Wait for vLLM server to be ready ─────────────────────────────────────────
echo "[launcher] Waiting for vLLM server (server_ready file, timeout=20 min)..."
READY_FILE="${OUTPUT_DIR}/server_ready"
DEADLINE=$(( $(date +%s) + 1200 ))   # 20 min
while [ ! -f "${READY_FILE}" ]; do
    if ! kill -0 "${RESAMPLE_SERVER_PID}" 2>/dev/null; then
        echo "[launcher] ERROR: resample_server exited before signalling ready!"
        exit 1
    fi
    if [ "$(date +%s)" -gt "${DEADLINE}" ]; then
        echo "[launcher] ERROR: timed out waiting for vLLM server."
        kill "${RESAMPLE_SERVER_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 10
done
echo "[launcher] vLLM server ready. Starting training..."
date

# ── Launch training (foreground, accelerate ZeRO-3 CPU-offload) ──────────────
/scratch/wzhao20/conda_envs/AKDA1/bin/accelerate launch \
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

# ── Graceful shutdown ─────────────────────────────────────────────────────────
echo "[launcher] Sending shutdown signal to resample server..."
echo '{"shutdown":true,"step":-1,"checkpoint_dir":"","model_name":"","seeds_and_questions":[]}' \
    > "${OUTPUT_DIR}/resample_request.json"

for i in $(seq 1 30); do
    if ! kill -0 "${RESAMPLE_SERVER_PID}" 2>/dev/null; then
        echo "[launcher] Resample server exited cleanly."
        break
    fi
    sleep 1
done
if kill -0 "${RESAMPLE_SERVER_PID}" 2>/dev/null; then
    echo "[launcher] Force-killing resample server (PID ${RESAMPLE_SERVER_PID})..."
    kill "${RESAMPLE_SERVER_PID}" 2>/dev/null || true
fi

echo "[launcher] All done."
exit "${TRAIN_RC}"
