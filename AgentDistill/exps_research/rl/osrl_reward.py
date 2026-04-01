"""
OS-RL reward functions.

R_task   : binary correctness of the agent's final answer.
R_sensitivity : measures how much each assistant thought depends on the
                preceding tool output.  Computed as:

    sensitivity_t = log p(thought_t | h_t, O_t)
                  - log p(thought_t | h_t, O_t_masked)

    R_sensitivity = mean_t( sensitivity_t )

where O_t is the tool-response content at step t and O_t_masked replaces
that content with a placeholder so the model cannot see the actual values.

No proxy student is required — both forward passes use the *teacher* model
itself (the one being trained).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helpers for message format
# --------------------------------------------------------------------------- #

def _get_text(content) -> str:
    """Unwrap smolagents content (list-of-dicts or plain string)."""
    if isinstance(content, list):
        parts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in content]
        return "".join(parts)
    return str(content or "")


def _flatten_for_lm(messages: List[Dict]) -> List[Dict]:
    """
    Convert smolagents roles to plain user/assistant/system roles,
    merging consecutive same-role messages.

    tool-call   → assistant
    tool-response → user
    """
    role_map = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool-call": "assistant",
        "tool-response": "user",
    }
    result: List[Dict] = []
    for msg in messages:
        role = role_map.get(msg["role"], "user")
        text = _get_text(msg["content"])
        if result and result[-1]["role"] == role:
            result[-1]["content"] += "\n" + text
        else:
            result.append({"role": role, "content": text})
    return result


def _find_sensitivity_steps(
    messages: List[Dict],
    mask_placeholder: str,
) -> List[Tuple[List[Dict], List[Dict], str]]:
    """
    Scan a trajectory and return a list of
        (context_with_output, context_masked_output, assistant_text)
    for every (tool-response → assistant) transition.
    """
    steps: List[Tuple[List[Dict], List[Dict], str]] = []
    for i, msg in enumerate(messages):
        if msg["role"] != "tool-response":
            continue
        if i + 1 >= len(messages):
            continue
        next_msg = messages[i + 1]
        if next_msg["role"] != "assistant":
            continue

        assistant_text = _get_text(next_msg["content"]).strip()
        if not assistant_text:
            continue

        # context *including* the tool-response (model sees O_t)
        ctx_with = _flatten_for_lm(messages[: i + 1])
        # context with the tool-response content replaced
        masked_messages = list(messages[: i]) + [
            {"role": "tool-response", "content": mask_placeholder}
        ]
        ctx_masked = _flatten_for_lm(masked_messages)

        steps.append((ctx_with, ctx_masked, assistant_text))
    return steps


# --------------------------------------------------------------------------- #
# Core log-prob computation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _conditional_log_prob(
    model,
    tokenizer,
    context_messages: List[Dict],
    response_text: str,
    device: torch.device,
    max_length: int = 8192,
) -> float:
    """
    Compute  sum_t log π(token_t | context, token_{<t})
    over the tokens in `response_text`, given `context_messages` as prefix.

    Returns 0.0 on any error (sequence too long, empty response, etc.).
    """
    if not response_text.strip():
        return 0.0

    # Tokenise context (with generation prompt so the model sees the right BOS)
    # apply_chat_template may return a BatchEncoding or a plain Tensor depending
    # on the transformers version; normalise to Tensor explicitly.
    _ctx = tokenizer.apply_chat_template(
        context_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ctx_ids = (_ctx["input_ids"] if hasattr(_ctx, "keys") else _ctx).to(device)  # (1, L_ctx)

    resp_ids = tokenizer(
        response_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)                                   # (1, L_resp)

    if resp_ids.shape[1] == 0:
        return 0.0

    full_ids = torch.cat([ctx_ids, resp_ids], dim=1)         # (1, L_ctx+L_resp)
    if full_ids.shape[1] > max_length:
        return 0.0

    outputs = model(input_ids=full_ids)
    logits = outputs.logits                                   # (1, L, V)

    resp_start = ctx_ids.shape[1]
    # Shift: logits[resp_start-1] predicts token[resp_start]
    resp_logits = logits[:, resp_start - 1 : -1, :]          # (1, L_resp, V)
    resp_targets = full_ids[:, resp_start:]                   # (1, L_resp)

    log_probs = F.log_softmax(resp_logits.float(), dim=-1)
    token_lps = log_probs.gather(-1, resp_targets.unsqueeze(-1)).squeeze(-1)
    return float(token_lps.sum())


# --------------------------------------------------------------------------- #
# Public reward functions
# --------------------------------------------------------------------------- #

def compute_task_reward(trajectory: Dict) -> float:
    """
    R_task = 1.0 if the agent's generated_answer matches true_answer, else 0.0.
    Uses the existing `is_correct` field when available (written by score_answers).
    """
    if "is_correct" in trajectory:
        return 1.0 if trajectory["is_correct"] else 0.0

    gen = str(trajectory.get("generated_answer", "")).strip()
    true = str(trajectory.get("true_answer", "")).strip()
    if not gen or not true:
        return 0.0
    # Exact-match fallback (scorer should have set is_correct already)
    return 1.0 if gen == true else 0.0


def compute_sensitivity_reward(
    model,
    tokenizer,
    trajectory: Dict,
    device: torch.device,
    mask_placeholder: str = "[MASKED_OBSERVATION]",
    max_steps: int = 5,
    max_length: int = 8192,
) -> float:
    """
    R_sensitivity for one trajectory.

    Args:
        model: the LoRA-wrapped training model (in eval mode, no grad).
        tokenizer: corresponding tokenizer.
        trajectory: one entry from the collected JSONL (with log_data.messages).
        device: torch device.
        mask_placeholder: text to replace tool-response content with.
        max_steps: maximum tool-call steps to average over.
        max_length: skip steps whose full sequence exceeds this token count.

    Returns:
        float — mean sensitivity across observed tool-call steps.
                0.0 if no tool-call steps exist.
    """
    messages = (trajectory.get("log_data") or {}).get("messages", [])
    if not messages:
        return 0.0

    steps = _find_sensitivity_steps(messages, mask_placeholder)
    if not steps:
        return 0.0

    sensitivities: List[float] = []
    for ctx_with, ctx_masked, assistant_text in steps[:max_steps]:
        lp_with = _conditional_log_prob(
            model, tokenizer, ctx_with, assistant_text, device, max_length
        )
        lp_masked = _conditional_log_prob(
            model, tokenizer, ctx_masked, assistant_text, device, max_length
        )
        sensitivities.append(lp_with - lp_masked)

    return float(np.mean(sensitivities)) if sensitivities else 0.0


def compute_rewards(
    model,
    tokenizer,
    trajectories: List[Dict],
    device: torch.device,
    lambda_sensitivity: float = 0.1,
    mask_placeholder: str = "[MASKED_OBSERVATION]",
    sensitivity_max_steps: int = 5,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute (total_rewards, task_rewards, sensitivity_rewards) for a batch.

    Returns three parallel lists of floats.
    """
    model.eval()
    r_task, r_sens, r_total = [], [], []
    for traj in trajectories:
        rt = compute_task_reward(traj)
        rs = compute_sensitivity_reward(
            model, tokenizer, traj, device,
            mask_placeholder=mask_placeholder,
            max_steps=sensitivity_max_steps,
        )
        r_task.append(rt)
        r_sens.append(rs)
        r_total.append(rt + lambda_sensitivity * rs)
    return r_total, r_task, r_sens
