"""
message_utils.py — Message preprocessing for GRPO training.

Handles the raw AgentDistill JSONL message format and converts it
into forms suitable for tokenisation and log-prob computation.

Raw message format (from smolagents):
    [
        {"role": "system",        "content": {"type": "text", "text": "..."}},
        {"role": "user",          "content": {"type": "text", "text": "..."}},
        {"role": "assistant",     "content": {"type": "text", "text": "Thought: ...\\nCode: ..."}},
        {"role": "tool-call",     "content": {"type": "text", "text": "Calling tools: ..."}},
        {"role": "tool-response", "content": {"type": "text", "text": "Observation: ..."}},
        ...
    ]

After clean_messages (mirroring SFT preprocessing):
    [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "question"},
        {"role": "assistant", "content": "Thought: ...\\nCode: ...<end_code>"},
        {"role": "user",      "content": "Observation: ..."},   ← former tool-response
        {"role": "assistant", "content": "Thought: ..."},
        ...
    ]
"""

import re
import copy
import logging
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

MASKED_OBS_TEXT = "[MASKED_OBSERVATION]"


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def extract_text(content) -> str:
    """
    Safely extract plain text from a message content field.
    Handles: str, dict {"type":"text","text":"..."}, list of such dicts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", str(content))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def normalise_message(msg: dict) -> dict:
    """Return a copy of msg with content as a plain string."""
    return {"role": msg["role"], "content": extract_text(msg.get("content", ""))}


# ---------------------------------------------------------------------------
# Message cleaning (mirrors preprocess.py clean_messages)
# ---------------------------------------------------------------------------

def clean_messages_for_training(raw_messages: List[dict]) -> Optional[List[dict]]:
    """
    Convert raw smolagents messages to clean [system, user, assistant, ...]
    format suitable for tokenisation.

    Reuses the existing smolagents + AgentDistill utilities so the format
    is identical to what the SFT pipeline produces.

    Returns None if cleaning fails (e.g., malformed messages).
    """
    try:
        from smolagents.models import remove_tool_call_from_messages, get_clean_message_list

        msgs = copy.deepcopy(raw_messages)

        # Normalise roles (MessageRole.XXX → "xxx")
        _ROLE_MAP = {
            "MessageRole.SYSTEM": "system",
            "MessageRole.USER": "user",
            "MessageRole.ASSISTANT": "assistant",
            "MessageRole.TOOL_CALL": "tool-call",
            "MessageRole.TOOL_RESPONSE": "tool-response",
        }
        for m in msgs:
            if m["role"] in _ROLE_MAP:
                m["role"] = _ROLE_MAP[m["role"]]

        # Remove tool-call messages (they're redundant; code is in assistant)
        msgs = remove_tool_call_from_messages(msgs)

        # Convert tool-response → user; flatten content to text
        msgs = get_clean_message_list(
            msgs,
            role_conversions={"tool-response": "user", "tool-call": "assistant"},
            flatten_messages_as_text=True,
        )

        # Remove trailing user message (no assistant response follows)
        if msgs and msgs[-1]["role"] == "user":
            msgs = msgs[:-1]

        # Remove <reference>...</reference> tags from user message
        if len(msgs) > 1:
            msgs[1]["content"] = re.sub(
                r"<reference>.*?</reference>", "", msgs[1]["content"], flags=re.DOTALL
            )

        return msgs

    except Exception as e:
        logger.debug(f"clean_messages_for_training failed: {e}")
        return None


# ---------------------------------------------------------------------------
# R_sens pair extraction
# ---------------------------------------------------------------------------

def get_rsens_pairs(
    cleaned_messages: List[dict],
) -> List[Tuple[List[dict], List[dict], dict]]:
    """
    Identify (context_with_obs, context_masked, target_thought) triples
    for computing R_sens.

    In the cleaned format the conversation alternates:
        [0] system
        [1] user   ← question (NOT an observation)
        [2] assistant  ← thought1 (no preceding obs → skip)
        [3] user   ← observation_1
        [4] assistant  ← thought2  ← R_sens pair here
        [5] user   ← observation_2
        [6] assistant  ← thought3  ← R_sens pair here
        ...

    For each assistant at index i ≥ 4 (i even, i ≥ 4) where
    cleaned_messages[i-1].role == "user" (observation), we yield:
        context_with:   cleaned_messages[0:i]   (includes obs at i-1)
        context_masked: cleaned_messages[0:i-1] + [masked_obs]
        target:         cleaned_messages[i]      (the assistant thought)

    Returns list of (context_with, context_masked, target) tuples.
    """
    pairs = []
    for i, msg in enumerate(cleaned_messages):
        if msg["role"] != "assistant":
            continue
        if i < 4:
            # First assistant (index 2) follows the question user, not an obs
            continue
        prev = cleaned_messages[i - 1]
        if prev["role"] != "user":
            continue
        # prev is an observation (index >= 3, and it's not the question at index 1)
        if i - 1 == 1:
            continue  # skip the question

        context_with = cleaned_messages[:i]
        masked_obs = {"role": "user", "content": MASKED_OBS_TEXT}
        context_masked = cleaned_messages[:i - 1] + [masked_obs]
        target = cleaned_messages[i]

        pairs.append((context_with, context_masked, target))

    return pairs


# ---------------------------------------------------------------------------
# Code extraction for diversity metric
# ---------------------------------------------------------------------------

def extract_code_blocks(cleaned_messages: List[dict]) -> List[str]:
    """
    Extract all Python code blocks from assistant messages.
    Returns a list of code strings (one per executed block).
    """
    code_blocks = []
    pattern = re.compile(r"```py\n(.*?)```", re.DOTALL)
    for msg in cleaned_messages:
        if msg["role"] != "assistant":
            continue
        content = msg.get("content", "")
        for match in pattern.finditer(content):
            code = match.group(1).strip()
            if code:
                code_blocks.append(code)
    return code_blocks


# ---------------------------------------------------------------------------
# Short system prompt (used during resampling / eval)
# ---------------------------------------------------------------------------

def get_short_system_prompt(python_only: bool = True) -> str:
    """
    Return the short system prompt used in SFT training.
    Mirrors preprocess.py's use of code_agent.yaml system_prompt_short.
    """
    import importlib
    import yaml
    try:
        prompt_data = yaml.safe_load(
            importlib.resources.files("smolagents.prompts")
            .joinpath("code_agent.yaml")
            .read_text()
        )
        template = prompt_data["system_prompt_short"]
        from smolagents import FinalAnswerTool
        from smolagents.agents import populate_template
        tools = {"final_answer": FinalAnswerTool()}
        return populate_template(template, variables={"tools": tools})
    except Exception as e:
        logger.warning(f"Could not load short system prompt: {e}")
        return "You are an expert assistant who can solve any task using code blobs."
