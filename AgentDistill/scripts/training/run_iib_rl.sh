#!/bin/bash
# run_iib_rl.sh — Input Invariance Breaking (IIB) RL on full MATH-500
# Branch: claude/serene-austin
#
# Reward:
#   R_total = R_task(τ_x) + λ_inv * R_inv(τ_x, τ_{x'})
#   R_inv   = mean normalised code-edit-distance between original and
#             augmented-question trajectories (range [0, 1])
#
# Augmentation: template-based instruction paraphrase (augment_utils.py)
#   - Same math problem, different framing
#   - No LLM call needed; template rotates each resample cycle
#
# Semi-online loop:
#   Every resample_every steps:
#     1. Save LoRA checkpoint
#     2. Collect fresh ORIG trajectories (n_resample_questions questions)
#     3. Collect fresh AUG trajectories (same questions, next template)
#     4. FIFO-replace orig pool; update aug pool
#
# Usage:
#   # Default run:
#   bash scripts/training/run_iib_rl.sh
#
#   # Custom lambda_inv:
#   LAMBDA_INV=2.0 bash scripts/training/run_iib_rl.sh
#
#   # Resume from checkpoint:
#   RESUME_FROM_CHECKPOINT=.../checkpoint-step200 \
#   INITIAL_STEP=200 \
#   bash scripts/training/run_iib_rl.sh

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
# trajectory_dir: flat directory of *_scored.jsonl from the pre-collected pool
# pilot_question_json: all 500 questions in MATH-500 format (for resampling universe)
TRAJ_DIR="${TRAJ_DIR:-/scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test/evaluations}"
PILOT_QUESTION_JSON="${PILOT_QUESTION_JSON:-$TRAJ_DIR/../pilot_questions.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/training_outputs/qwen3-32B/iib_rl}"
ACCEL_CONFIG="$ROOT_DIR/exps_research/mp_configs/accel_ds3_4gpu.yaml"

# ---- Hyperparameters ----
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
LAMBDA_INV="${LAMBDA_INV:-1.0}"        # divergence reward weight
NOISE_OPS="${NOISE_OPS:-2}"            # random noise ops per question (0=off)
LR="${LR:-1e-5}"
KL_COEFF="${KL_COEFF:-0.01}"
LORA_R="${LORA_R:-32}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:-1000}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
N_TRAJS="${N_TRAJS:-8}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
RESAMPLE_EVERY="${RESAMPLE_EVERY:-50}"
N_RESAMPLE_Q="${N_RESAMPLE_Q:-100}"           # questions per seed per cycle
SEEDS_PER_RESAMPLE="${SEEDS_PER_RESAMPLE:-5}" # 5 seeds × 100 q = 500 q/cycle
RESAMPLE_SEEDS="${RESAMPLE_SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56}"
MAX_AGENT_STEPS="${MAX_AGENT_STEPS:-5}"
QUALITY_MIN_ACC="${QUALITY_MIN_ACC:-0.30}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
INITIAL_STEP="${INITIAL_STEP:-0}"

mkdir -p "$OUTPUT_DIR"
RUN_LOG="$OUTPUT_DIR/run_iib_rl.log"

echo "=== IIB RL ===" | tee "$RUN_LOG"
date | tee -a "$RUN_LOG"
echo "  traj_dir:             $TRAJ_DIR"             | tee -a "$RUN_LOG"
echo "  pilot_question_json:  $PILOT_QUESTION_JSON"  | tee -a "$RUN_LOG"
echo "  output_dir:           $OUTPUT_DIR"           | tee -a "$RUN_LOG"
echo "  lambda_inv:           $LAMBDA_INV"           | tee -a "$RUN_LOG"
echo "  resample_every:       $RESAMPLE_EVERY steps" | tee -a "$RUN_LOG"
echo "  n_resample_questions: $N_RESAMPLE_Q"         | tee -a "$RUN_LOG"
echo "  resume_ckpt:          ${RESUME_FROM_CHECKPOINT:-(none)}" | tee -a "$RUN_LOG"

# ---- Preflight ----
"$PYTHON_BIN" - <<'PY'
import torch, transformers, peft, accelerate
print(f"torch={torch.__version__}  transformers={transformers.__version__}")
print(f"peft={peft.__version__}  accelerate={accelerate.__version__}")
print(f"GPUs: {torch.cuda.device_count()}")
PY

echo "" | tee -a "$RUN_LOG"
echo "Launching IIB RL..." | tee -a "$RUN_LOG"
date | tee -a "$RUN_LOG"

RESUME_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    RESUME_ARGS=(
        --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
        --initial_step           "$INITIAL_STEP"
    )
fi

"$ACCELERATE_BIN" launch \
    --config_file "$ACCEL_CONFIG" \
    -m exps_research.rl_training.train_iib_rl \
    --trajectory_dir      "$TRAJ_DIR" \
    --pilot_question_json "$PILOT_QUESTION_JSON" \
    --model_name          "$MODEL_NAME" \
    --lambda_inv          "$LAMBDA_INV" \
    --lr                  "$LR" \
    --kl_coeff            "$KL_COEFF" \
    --lora_r              "$LORA_R" \
    --max_length          "$MAX_LENGTH" \
    --num_epochs          "$NUM_EPOCHS" \
    --max_steps           "$MAX_STEPS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --n_trajs_per_question "$N_TRAJS" \
    --checkpoint_every    "$CHECKPOINT_EVERY" \
    --resample_every       "$RESAMPLE_EVERY" \
    --n_resample_questions "$N_RESAMPLE_Q" \
    --seeds_per_resample   "$SEEDS_PER_RESAMPLE" \
    --resample_seeds       $RESAMPLE_SEEDS \
    --max_agent_steps     "$MAX_AGENT_STEPS" \
    --quality_min_acc     "$QUALITY_MIN_ACC" \
    --noise_ops           "$NOISE_OPS" \
    --output_dir          "$OUTPUT_DIR" \
    --log_every           5 \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "$RUN_LOG"

echo "=== IIB RL complete ===" | tee -a "$RUN_LOG"
date | tee -a "$RUN_LOG"
