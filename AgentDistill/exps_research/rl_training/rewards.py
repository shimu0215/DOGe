"""
rewards.py — Reward computation for OS-RL and Diversity-RL.

R_task:      Binary correctness from pre-scored JSONL (no model needed).
R_sens:      Output-sensitivity reward.  Requires teacher model forward pass.
R_diversity: Trajectory diversity reward. Requires only code extraction.
"""

import re
import math
import copy
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F

from .message_utils import (
    clean_messages_for_training,
    get_rsens_pairs,
    extract_code_blocks,
    MASKED_OBS_TEXT,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# R_task
# ===========================================================================

def compute_r_task(group: List[dict]) -> List[float]:
    """
    Extract binary correctness reward from pre-scored trajectory entries.
    score=True → 1.0,  score=False → 0.0
    """
    rewards = []
    for entry in group:
        s = entry.get("score", False)
        rewards.append(1.0 if s else 0.0)
    return rewards


# ===========================================================================
# R_sens
# ===========================================================================

@torch.no_grad()
def compute_r_sens(
    model,
    tokenizer,
    group: List[dict],
    max_target_tokens: int = 512,
    device: Optional[torch.device] = None,
) -> List[float]:
    """
    Compute Output-Sensitivity reward for a group of trajectories.

    For each assistant step t that follows an observation (tool-response):
        sens_t = log p(thought_t | context_with_obs_t)
               - log p(thought_t | context_with_MASKED_obs_t)

    R_sens(trajectory) = mean_t(sens_t)

    Single-step trajectories (no preceding observation) get R_sens = 0.0.

    Args:
        model:      HuggingFace CausalLM (already on correct devices).
        tokenizer:  Corresponding tokenizer.
        group:      List of trajectory dicts (same question).
        max_target_tokens: Max tokens in the target thought to evaluate.
        device:     Target device for tensors (auto-detected if None).

    Returns:
        List of R_sens values, one per trajectory.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    r_sens_values = []

    for entry in group:
        raw_messages = entry.get("log_data", {}).get("messages", [])
        cleaned = clean_messages_for_training(raw_messages)
        if cleaned is None or len(cleaned) < 5:
            # Single-step or failed to clean → R_sens = 0
            r_sens_values.append(0.0)
            continue

        pairs = get_rsens_pairs(cleaned)
        if not pairs:
            r_sens_values.append(0.0)
            continue

        step_scores = []
        for ctx_with, ctx_masked, target in pairs:
            lp_with = _compute_conditional_log_prob(
                model, tokenizer, ctx_with, target,
                max_target_tokens=max_target_tokens, device=device
            )
            lp_masked = _compute_conditional_log_prob(
                model, tokenizer, ctx_masked, target,
                max_target_tokens=max_target_tokens, device=device
            )
            if lp_with is not None and lp_masked is not None:
                step_scores.append(lp_with - lp_masked)

        if step_scores:
            r_sens_values.append(float(np.mean(step_scores)))
        else:
            r_sens_values.append(0.0)

    return r_sens_values


def _compute_conditional_log_prob(
    model,
    tokenizer,
    context_messages: List[dict],
    target_message: dict,
    max_target_tokens: int,
    device: torch.device,
    max_context_tokens: int = 1024,
) -> Optional[float]:
    """
    Compute mean log p(target_tokens | context) using teacher forcing.

    Strategy:
      1. Tokenise context_messages → ctx_ids  (length L_ctx)
      2. Tokenise context + target → full_ids  (length L_full)
      3. target_ids = full_ids[L_ctx:]         (length L_tgt)
      4. Forward pass on full_ids → logits
      5. log_p = mean( log_softmax(logits)[L_ctx-1 : L_ctx-1+L_tgt, target_ids] )

    Context is truncated from the LEFT to max_context_tokens to avoid OOM
    (no flash_attn → attention is O(L²)).

    Returns None if tokenisation fails or target is empty.
    """
    try:
        # Build context-only token ids
        ctx_ids = _apply_template(tokenizer, context_messages, add_generation_prompt=True)
        if ctx_ids is None:
            return None

        # Build full (context + target) token ids
        full_messages = context_messages + [target_message]
        full_ids = _apply_template(tokenizer, full_messages, add_generation_prompt=False)
        if full_ids is None:
            return None

        L_ctx = ctx_ids.shape[1]
        L_full = full_ids.shape[1]
        L_tgt = L_full - L_ctx

        if L_tgt <= 0:
            return None

        # Truncate target if too long
        if L_tgt > max_target_tokens:
            full_ids = full_ids[:, :L_ctx + max_target_tokens]
            L_tgt = max_target_tokens
            L_full = L_ctx + L_tgt

        # Truncate context from the left if needed (keep target intact)
        if L_full > max_context_tokens + L_tgt:
            keep = max_context_tokens + L_tgt
            trim = L_full - keep
            full_ids = full_ids[:, trim:]
            L_ctx = max(1, L_ctx - trim)
            L_full = full_ids.shape[1]
            L_tgt = L_full - L_ctx

        full_ids = full_ids.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=full_ids)
        logits = outputs.logits  # [1, L_full, vocab]

        # We predict token at position i using logits at position i-1
        # Target tokens are at positions L_ctx .. L_full-1
        # Logits we use: positions L_ctx-1 .. L_full-2
        target_ids = full_ids[0, L_ctx:L_full]           # [L_tgt]
        pred_logits = logits[0, L_ctx - 1 : L_full - 1, :]  # [L_tgt, vocab]

        log_probs = F.log_softmax(pred_logits.float(), dim=-1)  # [L_tgt, vocab]
        token_log_probs = log_probs[range(L_tgt), target_ids.cpu()]  # [L_tgt]

        return float(token_log_probs.mean().item())

    except Exception as e:
        logger.debug(f"_compute_conditional_log_prob failed: {e}")
        return None


def _apply_template(tokenizer, messages: List[dict], add_generation_prompt: bool):
    """
    Apply Qwen chat template and return token ids as [1, L] tensor.
    Returns None on failure.
    """
    try:
        # Ensure content is plain string
        clean_msgs = [
            {"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else str(m["content"])}
            for m in messages
        ]
        ids = tokenizer.apply_chat_template(
            clean_msgs,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        return ids
    except Exception as e:
        logger.debug(f"_apply_template failed: {e}")
        return None


# ===========================================================================
# R_diversity
# ===========================================================================

def compute_r_diversity(group: List[dict]) -> List[float]:
    """
    Compute trajectory diversity reward for a group sharing the same question.

    For each trajectory i, its diversity reward is the mean normalised edit
    distance to all other trajectories in the group.

    Edit distance is computed at the line level over the concatenated Python
    code blocks of each trajectory.

    Returns list of diversity values in [0, 1].
    """
    if len(group) <= 1:
        return [0.0] * len(group)

    # Extract code string for each trajectory
    code_strings = []
    for entry in group:
        raw_messages = entry.get("log_data", {}).get("messages", [])
        cleaned = clean_messages_for_training(raw_messages)
        if cleaned is None:
            code_strings.append("")
            continue
        blocks = extract_code_blocks(cleaned)
        code_strings.append("\n".join(blocks))

    n = len(code_strings)
    rewards = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            d = _normalised_edit_distance(code_strings[i], code_strings[j])
            dists.append(d)
        rewards.append(float(np.mean(dists)) if dists else 0.0)

    return rewards


def _normalised_edit_distance(a: str, b: str) -> float:
    """
    Line-level Levenshtein edit distance normalised by max(len(a_lines), len(b_lines)).
    Returns value in [0, 1].
    """
    a_lines = a.split("\n")
    b_lines = b.split("\n")
    la, lb = len(a_lines), len(b_lines)
    if la == 0 and lb == 0:
        return 0.0
    dist = _levenshtein(a_lines, b_lines)
    return dist / max(la, lb)


def _levenshtein(seq_a: List[str], seq_b: List[str]) -> int:
    """Standard dynamic-programming Levenshtein on lists of strings."""
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[n]


# ===========================================================================
# Combined reward helpers
# ===========================================================================

def compute_total_rewards_os_rl(
    model,
    tokenizer,
    group: List[dict],
    lambda_sens: float = 0.1,
    device=None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    R_total = R_task + lambda_sens * R_sens

    Returns: (r_total, r_task, r_sens)
    """
    r_task = compute_r_task(group)
    r_sens = compute_r_sens(model, tokenizer, group, device=device)
    r_total = [rt + lambda_sens * rs for rt, rs in zip(r_task, r_sens)]
    return r_total, r_task, r_sens


def compute_total_rewards_div_rl(
    group: List[dict],
    lambda_div: float = 0.5,
) -> Tuple[List[float], List[float], List[float]]:
    """
    R_total = R_task + lambda_div * R_diversity

    Returns: (r_total, r_task, r_div)
    """
    r_task = compute_r_task(group)
    r_div = compute_r_diversity(group)
    r_total = [rt + lambda_div * rd for rt, rd in zip(r_task, r_div)]
    return r_total, r_task, r_div
