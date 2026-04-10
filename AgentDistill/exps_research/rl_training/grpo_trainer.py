"""
grpo_trainer.py — Offline GRPO training core.

Implements:
  - Log-probability computation over cleaned message sequences
  - Group-normalised advantage estimation
  - Policy gradient loss (optionally with per-token KL penalty vs. base model)
  - Gradient accumulation helper

Design notes:
  - Works with LoRA models where the frozen base = implicit reference.
    KL penalty is estimated by temporarily disabling LoRA adapters.
  - DeepSpeed ZeRO-3 compatible: relies only on standard forward passes.
  - Tokenisation follows the same Qwen chat-template convention as finetune_sft.py.

Memory strategy:
  - compute_trajectory_log_probs uses F.cross_entropy on ONLY the assistant-token
    slice of logits, avoiding materialising a full [L, vocab] fp32 tensor.
  - GRPOTrainer.step processes ONE trajectory at a time and calls backward()
    immediately after each, so at most one gradient graph exists at any moment.
  - torch.cuda.empty_cache() is called between trajectories.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F

from .message_utils import clean_messages_for_training

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def compute_group_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
    """
    Group-normalise a list of rewards to get advantages.
    A_i = (R_i - mean(R)) / (std(R) + eps)
    """
    r = np.array(rewards, dtype=np.float32)
    mean_r = r.mean()
    std_r = r.std() + eps
    return list((r - mean_r) / std_r)


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def tokenise_conversation(
    tokenizer,
    messages: List[dict],
) -> Optional[torch.Tensor]:
    """
    Apply the chat template and return a 1-D LongTensor of token ids.
    Content must be plain strings already (post-clean_messages).
    Returns None on failure.
    """
    try:
        clean_msgs = [
            {"role": m["role"],
             "content": m["content"] if isinstance(m["content"], str) else str(m["content"])}
            for m in messages
        ]
        ids = tokenizer.apply_chat_template(
            clean_msgs,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )  # [1, L]
        return ids.squeeze(0)  # [L]
    except Exception as e:
        logger.debug(f"tokenise_conversation failed: {e}")
        return None


def find_assistant_token_ranges(
    input_ids: torch.Tensor,
    tokenizer,
) -> List[Tuple[int, int]]:
    """
    Find [start, end) ranges of assistant response tokens in input_ids.

    We look for the token sequence corresponding to "<|im_start|>assistant\\n"
    (Qwen chat template) and treat everything until the next "<|im_end|>" as
    assistant tokens.

    Returns list of (start_excl, end_excl) index pairs.
    start_excl points to the first token AFTER the role header (i.e., the
    first content token).  end_excl points to the <|im_end|> token (excluded).
    """
    # Tokenise the role-start template
    try:
        header_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if not end_id:
            return []
        end_token = end_id[0]
    except Exception:
        return []

    ids = input_ids.tolist()
    H = len(header_ids)
    ranges = []
    i = 0
    while i <= len(ids) - H:
        if ids[i : i + H] == header_ids:
            start = i + H  # first content token
            # find the closing <|im_end|>
            end = start
            while end < len(ids) and ids[end] != end_token:
                end += 1
            if end > start:
                ranges.append((start, end))
            i = end + 1
        else:
            i += 1
    return ranges


# ---------------------------------------------------------------------------
# Per-trajectory log-prob computation  (memory-efficient)
# ---------------------------------------------------------------------------

def compute_trajectory_log_probs(
    model,
    tokenizer,
    cleaned_messages: List[dict],
    device: torch.device,
    max_length: int = 4096,
) -> Optional[torch.Tensor]:
    """
    Compute log p_θ(assistant_token_t | all_preceding_tokens) for every
    assistant token in the trajectory.

    Memory optimisation:
      Instead of log_softmax(logits[0].float())  →  [L, vocab] fp32 (huge),
      we use F.cross_entropy on the small assistant-token slice [n_asst, vocab],
      which avoids materialising the full vocabulary matrix for all L positions.

    Returns a 1-D FloatTensor of per-token log probs (assistant tokens only).
    Returns None if tokenisation fails or no assistant tokens found.
    """
    input_ids = tokenise_conversation(tokenizer, cleaned_messages)
    if input_ids is None:
        return None

    # Truncate from the LEFT if needed, so assistant tokens at the END are preserved
    if input_ids.shape[0] > max_length:
        input_ids = input_ids[-max_length:]

    ranges = find_assistant_token_ranges(input_ids, tokenizer)
    if not ranges:
        return None

    input_ids = input_ids.unsqueeze(0).to(device)  # [1, L]

    logits = model(input_ids=input_ids).logits  # [1, L, V]  bf16

    # Process only the assistant-token positions to avoid materialising [L, V] in fp32
    all_log_probs = []
    for start, end in ranges:
        if start == 0:
            continue
        targets = input_ids[0, start:end]                   # [n]
        pred_logits = logits[0, start - 1 : end - 1, :]    # [n, V] — view of logits

        # F.cross_entropy: numerically stable, gradient preserved through logits,
        # and only materialises softmax for [n, V] not [L, V].
        token_lp = -F.cross_entropy(
            pred_logits.float(), targets, reduction="none"
        )  # [n]
        all_log_probs.append(token_lp)

    if not all_log_probs:
        return None

    return torch.cat(all_log_probs)  # 1-D tensor, has grad


# ---------------------------------------------------------------------------
# Reference-model log-prob (for KL penalty)
# ---------------------------------------------------------------------------

def _ref_log_prob(
    model,
    tokenizer,
    cleaned_messages: List[dict],
    device: torch.device,
    max_length: int,
) -> Optional[torch.Tensor]:
    """
    Compute mean log prob under the reference model (base, no LoRA).
    Temporarily disables LoRA adapter layers if present.
    Returns scalar, detached.
    """
    has_lora = hasattr(model, "disable_adapter_layers")
    try:
        if has_lora:
            model.disable_adapter_layers()
        with torch.no_grad():
            lp = compute_trajectory_log_probs(
                model, tokenizer, cleaned_messages, device, max_length
            )
        if lp is None:
            return None
        return lp.mean()
    except Exception as e:
        logger.debug(f"_ref_log_prob failed: {e}")
        return None
    finally:
        if has_lora:
            model.enable_adapter_layers()


# ---------------------------------------------------------------------------
# GRPOTrainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """
    Thin wrapper around the GRPO loss for use in a custom Accelerate training loop.

    Memory strategy: processes ONE trajectory at a time, calling
    accelerator.backward() immediately after each, so at most one gradient
    graph lives in memory simultaneously.  torch.cuda.empty_cache() is called
    between trajectories to release any fragmented allocations.

    Usage:
        trainer = GRPOTrainer(model, tokenizer, optimizer, accelerator, config)
        for step, batch_groups in enumerate(loader):
            reward_fn(batch_groups) → rewards_per_group
            info = trainer.step(batch_groups, rewards_per_group, step)
    """

    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        accelerator,
        config: Dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.config = config

        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
        self.kl_coeff = config.get("kl_coeff", 0.0)
        self.max_length = config.get("max_length", 4096)
        self.clip_ratio = config.get("clip_ratio", 0.2)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        self._accum_loss = 0.0
        self._accum_steps = 0
        self._optimizer_steps = 0

    @property
    def device(self):
        try:
            return self.accelerator.device
        except Exception:
            return next(self.model.parameters()).device

    def step(
        self,
        batch_groups: List[List[dict]],
        rewards_per_group: List[List[float]],
        global_step: int,
    ) -> Dict:
        """
        Run one GRPO accumulation step.

        Processes each trajectory INDIVIDUALLY: forward → backward (frees graph)
        → empty_cache.  This limits peak GPU memory to one trajectory's
        gradient graph at a time, regardless of group size.

        Loss scaling:
          loss_i = (-A_i * mean_log_p_i + kl * kl_i) / (n_total * grad_accum)
        so gradients are equivalent to mean over the full group × accum window.

        Calls optimizer.step() every `gradient_accumulation_steps` calls.
        Returns dict with loss info (averaged over the accumulation window).
        """
        info = {"loss": 0.0, "optimizer_step": False}

        # Compute advantages per group (pure Python, no GPU)
        adv_per_group = [compute_group_advantages(r) for r in rewards_per_group]

        # Denominator for loss scaling: total trajectories × grad_accum steps
        n_total = sum(len(g) for g in batch_groups)
        if n_total == 0:
            return info
        scale = 1.0 / (n_total * self.gradient_accumulation_steps)

        step_loss_sum = 0.0
        step_valid = 0

        for group, advantages in zip(batch_groups, adv_per_group):
            for entry, adv in zip(group, advantages):
                raw_messages = entry.get("log_data", {}).get("messages", [])
                cleaned = clean_messages_for_training(raw_messages)
                if cleaned is None:
                    continue

                # ---- forward pass for this single trajectory (with grad) ----
                log_probs = compute_trajectory_log_probs(
                    self.model, self.tokenizer, cleaned,
                    self.device, self.max_length,
                )
                if log_probs is None:
                    continue

                mean_log_p = log_probs.mean()  # scalar, has grad

                # ---- optional KL term (detached → no grad through kl_term) ----
                kl_term = torch.tensor(0.0, device=self.device)
                if self.kl_coeff > 0.0:
                    ref_lp = _ref_log_prob(
                        self.model, self.tokenizer, cleaned,
                        self.device, self.max_length,
                    )
                    if ref_lp is not None:
                        # KL ≈ log p_θ - log p_ref  (per-token average, detached)
                        kl_term = mean_log_p.detach() - ref_lp

                loss_i = (-float(adv) * mean_log_p + self.kl_coeff * kl_term) * scale

                # ---- backward IMMEDIATELY — frees this trajectory's graph ----
                self.accelerator.backward(loss_i)

                step_loss_sum += loss_i.item() / scale  # unscaled for logging
                step_valid += 1

                # Release fragmented allocations before next trajectory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if step_valid == 0:
            return info

        self._accum_loss += step_loss_sum / step_valid
        self._accum_steps += 1

        # Optimizer step every gradient_accumulation_steps
        if (self._accum_steps % self.gradient_accumulation_steps) == 0:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._optimizer_steps += 1
            info["optimizer_step"] = True
            info["loss"] = self._accum_loss / self.gradient_accumulation_steps
            self._accum_loss = 0.0

        return info
