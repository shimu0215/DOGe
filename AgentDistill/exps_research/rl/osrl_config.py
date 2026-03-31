"""
OS-RL configuration dataclass.
All hyperparameters for the Output-Sensitivity RL training loop.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OSRLConfig:
    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model_name: str = "Qwen/Qwen3-32B"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.0

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    data_path: str = ""          # path to math JSON dataset
    output_dir: str = ""         # where to save checkpoints + rollouts

    # ------------------------------------------------------------------ #
    # Rollout collection (vLLM)
    # ------------------------------------------------------------------ #
    num_rollouts_per_problem: int = 4   # G in GRPO
    rollout_batch_size: int = 32        # problems per RL iteration
    max_agent_steps: int = 5
    max_tokens: int = 1024
    rollout_temperature: float = 0.7
    rollout_top_p: float = 0.8
    rollout_seed: int = 42
    vllm_tp_size: int = 4
    vllm_port: int = 8000
    vllm_gpu_util: float = 0.85
    parallel_workers: int = 4
    vllm_startup_timeout: int = 1800    # seconds to wait for vLLM to start

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    lambda_sensitivity: float = 0.1    # weight of R_sensitivity in total reward
    sensitivity_mask_token: str = "[MASKED_OBSERVATION]"
    sensitivity_max_steps: int = 5      # max tool-call steps to average over

    # ------------------------------------------------------------------ #
    # GRPO / training
    # ------------------------------------------------------------------ #
    learning_rate: float = 1e-5
    num_rl_iterations: int = 100        # total outer RL iterations
    grpo_epochs_per_iter: int = 1       # gradient steps per rollout batch
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    kl_coef: float = 0.01              # KL(policy || ref) penalty coefficient
    clip_eps: float = 0.2              # PPO-style clip on probability ratio
    grpo_eps: float = 1e-8             # advantage normalisation epsilon
    max_seq_length: int = 10240

    # ------------------------------------------------------------------ #
    # Logging / checkpointing
    # ------------------------------------------------------------------ #
    seed: int = 42
    save_every_n_iters: int = 10
    log_every_n_iters: int = 1
    resume_from_checkpoint: Optional[str] = None
