#!/bin/bash
# run_os_rl_online_pilot.sh — Semi-online OS-RL pilot (6779022 / gpu026)
# Branch: claude/可多拉
#
# Workflow:
#   1. Prepare 50-question pilot data (idempotent) + pilot_questions.json
#   2. Run semi-online OS-RL:
#        - Train for resample_every steps
#        - Save checkpoint
#        - Collect fresh trajectories via collect_unit.sh (vLLM tp=1, GPU3)
#        - Refresh pool, continue
#
# GPU layout:
#   Training : accelerate (4 GPUs, ZeRO-3 CPU offload, ~1 GB each)
#   vLLM     : tp=1, CUDA_VISIBLE_DEVICES=3 (~79 GB free on GPU3)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- Environment ----
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/wzhao20/conda_envs/AKDA1}"
export PATH="$CONDA_ENV_PREFIX/bin:${PATH:-}"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/wzhao20/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/wzhao20/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/wzhao20/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/wzhao20/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/wzhao20/torchinductor_cache}"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="$CONDA_ENV_PREFIX/bin/python"
ACCELERATE_BIN="$CONDA_ENV_PREFIX/bin/accelerate"

# ---- Paths ----
FULL_TRAJ_DIR="${FULL_TRAJ_DIR:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test/evaluations}"
PILOT_TRAJ_DIR="${PILOT_TRAJ_DIR:-$ROOT_DIR/training_outputs/qwen3-32B/os_rl_pilot/pilot_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/training_outputs/qwen3-32B/os_rl_online_pilot}"
ACCEL_CONFIG="$ROOT_DIR/exps_research/mp_configs/accel_ds3_4gpu.yaml"

# ---- Parameters ----
N_QUESTIONS="${N_QUESTIONS:-50}"
SEED_RANGE_START="${SEED_RANGE_START:-42}"
SEED_RANGE_END="${SEED_RANGE_END:-57}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
LAMBDA_SENS="${LAMBDA_SENS:-0.1}"
LR="${LR:-1e-5}"
KL_COEFF="${KL_COEFF:-0.01}"
LORA_R="${LORA_R:-32}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:-150}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
N_TRAJS="${N_TRAJS:-8}"
RESAMPLE_EVERY="${RESAMPLE_EVERY:-50}"    # checkpoint + resample every 50 steps
RESAMPLE_SEEDS="${RESAMPLE_SEEDS:-42 43 44}"
MAX_AGENT_STEPS="${MAX_AGENT_STEPS:-5}"

mkdir -p "$OUTPUT_DIR" "$PILOT_TRAJ_DIR"

RUN_LOG="$OUTPUT_DIR/run_os_rl_online_pilot.log"

echo "=== Semi-online OS-RL Pilot ===" | tee "$RUN_LOG"
date | tee -a "$RUN_LOG"

# ---- Step 1: Prepare pilot data + pilot_questions.json ----
if [ ! -f "$PILOT_TRAJ_DIR/pilot_questions.json" ]; then
    echo "Preparing pilot data..." | tee -a "$RUN_LOG"
    "$PYTHON_BIN" -m exps_research.rl_training.prepare_pilot_data \
        --traj_dir    "$FULL_TRAJ_DIR" \
        --output_dir  "$PILOT_TRAJ_DIR" \
        --n_questions "$N_QUESTIONS" \
        --seed_range  "$SEED_RANGE_START" "$SEED_RANGE_END" \
        --seed        0 \
        2>&1 | tee -a "$RUN_LOG"
    echo "Pilot data ready." | tee -a "$RUN_LOG"
else
    echo "Pilot data already exists, skipping preparation." | tee -a "$RUN_LOG"
fi

PILOT_QUESTION_JSON="$PILOT_TRAJ_DIR/pilot_questions.json"

# ---- Preflight ----
"$PYTHON_BIN" - <<'PY'
import torch, transformers, peft, accelerate
print(f"torch={torch.__version__}  transformers={transformers.__version__}")
print(f"peft={peft.__version__}  accelerate={accelerate.__version__}")
print(f"GPUs: {torch.cuda.device_count()}")
PY

# ---- Step 2: Launch semi-online training ----
echo "" | tee -a "$RUN_LOG"
echo "Launching semi-online OS-RL..." | tee -a "$RUN_LOG"
echo "  lambda_sens:     $LAMBDA_SENS"   | tee -a "$RUN_LOG"
echo "  resample_every:  $RESAMPLE_EVERY steps (also checkpoint cadence)" | tee -a "$RUN_LOG"
echo "  resample_seeds:  $RESAMPLE_SEEDS" | tee -a "$RUN_LOG"
echo "  vLLM:            tp=1, CUDA_VISIBLE_DEVICES=3" | tee -a "$RUN_LOG"
echo "  output_dir:      $OUTPUT_DIR"    | tee -a "$RUN_LOG"
date | tee -a "$RUN_LOG"

"$ACCELERATE_BIN" launch \
    --config_file "$ACCEL_CONFIG" \
    -m exps_research.rl_training.train_os_rl_online_pilot \
    --trajectory_dir      "$PILOT_TRAJ_DIR" \
    --pilot_question_json "$PILOT_QUESTION_JSON" \
    --seed_range          "$SEED_RANGE_START" "$SEED_RANGE_END" \
    --model_name          "$MODEL_NAME" \
    --lambda_sens         "$LAMBDA_SENS" \
    --lr                  "$LR" \
    --kl_coeff            "$KL_COEFF" \
    --lora_r              "$LORA_R" \
    --max_length          "$MAX_LENGTH" \
    --num_epochs          "$NUM_EPOCHS" \
    --max_steps           "$MAX_STEPS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --n_trajs_per_question "$N_TRAJS" \
    --resample_every      "$RESAMPLE_EVERY" \
    --resample_seeds      $RESAMPLE_SEEDS \
    --max_agent_steps     "$MAX_AGENT_STEPS" \
    --output_dir          "$OUTPUT_DIR" \
    --log_every           5 \
    2>&1 | tee -a "$RUN_LOG"

echo "=== Semi-online OS-RL pilot complete ===" | tee -a "$RUN_LOG"
date | tee -a "$RUN_LOG"
