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
# Per-trajectory log-prob computation
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

    logits = model(input_ids=input_ids).logits  # [1, L, V]
    log_probs = F.log_softmax(logits[0].float(), dim=-1)  # [L, V]

    # For each assistant token at position p, the prediction logit is at p-1
    all_log_probs = []
    for start, end in ranges:
        if start == 0:
            continue
        targets = input_ids[0, start:end]           # [n_tokens]
        preds   = log_probs[start - 1 : end - 1, :] # [n_tokens, V]
        token_lp = preds[range(len(targets)), targets.cpu()]
        all_log_probs.append(token_lp)

    if not all_log_probs:
        return None

    return torch.cat(all_log_probs)  # 1-D tensor


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def grpo_loss_for_group(
    model,
    tokenizer,
    group_trajectories: List[dict],
    advantages: List[float],
    device: torch.device,
    kl_coeff: float = 0.0,
    max_length: int = 4096,
    clip_ratio: float = 0.2,
) -> Optional[torch.Tensor]:
    """
    Compute GRPO policy-gradient loss for one question group.

    L = - mean_i [ A_i * mean_t(log π_θ(a_t|s_t)) ]
      + kl_coeff * mean_i [ KL(π_θ || π_ref) ]

    With LoRA, π_ref is approximated by temporarily disabling adapters.

    Returns scalar loss tensor (with grad), or None if all tokenisations fail.
    """
    losses = []

    for entry, adv in zip(group_trajectories, advantages):
        raw_messages = entry.get("log_data", {}).get("messages", [])
        cleaned = clean_messages_for_training(raw_messages)
        if cleaned is None:
            continue

        # ---- compute log probs under current policy ----
        log_probs = compute_trajectory_log_probs(
            model, tokenizer, cleaned, device, max_length=max_length
        )
        if log_probs is None:
            continue

        mean_log_p = log_probs.mean()  # scalar, has grad

        # ---- optional KL term (base model as ref) ----
        kl_term = torch.tensor(0.0, device=device)
        if kl_coeff > 0.0:
            ref_mean_log_p = _ref_log_prob(
                model, tokenizer, cleaned, device, max_length
            )
            if ref_mean_log_p is not None:
                # KL ≈ log p_θ - log p_ref (per token average)
                kl_term = mean_log_p.detach() - ref_mean_log_p  # scalar

        loss_i = -float(adv) * mean_log_p + kl_coeff * kl_term
        losses.append(loss_i)

    if not losses:
        return None

    return torch.stack(losses).mean()


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

    Usage:
        trainer = GRPOTrainer(model, tokenizer, optimizer, accelerator, config)
        for step, batch_groups in enumerate(loader):
            reward_fn(batch_groups) → rewards_per_group
            loss = trainer.step(batch_groups, rewards_per_group, step)
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

        Computes advantages, loss, backward pass.  Calls optimizer.step()
        every `gradient_accumulation_steps` calls.

        Returns dict with loss info (averaged over the accumulation window).
        """
        info = {"loss": 0.0, "optimizer_step": False}

        # Compute advantages per group
        adv_per_group = [compute_group_advantages(r) for r in rewards_per_group]

        # Accumulate losses over groups
        total_loss = None
        n_valid = 0

        for group, advantages in zip(batch_groups, adv_per_group):
            group_loss = grpo_loss_for_group(
                model=self.model,
                tokenizer=self.tokenizer,
                group_trajectories=group,
                advantages=advantages,
                device=self.device,
                kl_coeff=self.kl_coeff,
                max_length=self.max_length,
                clip_ratio=self.clip_ratio,
            )
            if group_loss is None:
                continue
            if total_loss is None:
                total_loss = group_loss
            else:
                total_loss = total_loss + group_loss
            n_valid += 1

        if total_loss is None or n_valid == 0:
            return info

        total_loss = total_loss / (n_valid * self.gradient_accumulation_steps)
        self.accelerator.backward(total_loss)

        self._accum_loss += total_loss.item() * self.gradient_accumulation_steps
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
