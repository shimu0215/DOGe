"""
Step-prefix evaluation.

For each correct trajectory of n steps, for each prefix of i steps (i=1..n),
build a provide_final_answer-style prompt and ask the model for an immediate
answer. Records per-step accuracy (over n_samples draws) and the logprob sum
for tokens inside \\boxed{} in each generation.
"""

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def _setup_path():
    this_dir = Path(__file__).resolve().parent
    for p in [str(this_dir.parent / "src"), str(this_dir.parent)]:
        if p not in sys.path:
            sys.path.insert(0, p)


_setup_path()

from smolagents import VLLMServerModel
from smolagents.agents import _extract_generation_logprobs, populate_template

from exps_research.unified_framework.math_utils.qwen_math_grader import math_equal
from exps_research.unified_framework.math_utils.qwen_math_parser import extract_answer

PROMPT_TEMPLATES = yaml.safe_load(
    importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
)


# ---------------------------------------------------------------------------
# Step boundary detection
# ---------------------------------------------------------------------------

def find_step_end_indices(messages: List[Dict]) -> List[int]:
    """Return exclusive-end index for each step.
    Each step ends at its tool-response (or error-response) message.
    """
    ends = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role in ("tool-response", "MessageRole.TOOL_RESPONSE"):
            ends.append(i + 1)
    return ends


def get_task_from_messages(messages: List[Dict]) -> str:
    """Extract raw task string from messages[1] (TaskStep: 'New task:\\n{task}')."""
    if len(messages) < 2:
        return ""
    content = messages[1].get("content", [])
    raw = content[0].get("text", "") if isinstance(content, list) and content else str(content)
    return raw.removeprefix("New task:\n")


# ---------------------------------------------------------------------------
# Prompt construction (mirrors MultiStepAgent.provide_final_answer)
# ---------------------------------------------------------------------------

def build_final_answer_messages(prefix_messages: List[Dict], task: str) -> List[Dict]:
    """Build the provide_final_answer prompt using a truncated message prefix."""
    pre = PROMPT_TEMPLATES["final_answer"]["pre_messages"]
    post_tmpl = PROMPT_TEMPLATES["final_answer"]["post_messages"]

    clean_task = task.split("\n\nIMPORTANT:")[0] if "\n\nIMPORTANT:" in task else task
    post = populate_template(post_tmpl, variables={"task": clean_task})

    return (
        [{"role": "system", "content": [{"type": "text", "text": pre}]}]
        + prefix_messages[1:]   # drop original system prompt
        + [{"role": "user",   "content": [{"type": "text", "text": post}]}]
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_math_answer(generated: str, gold: str) -> bool:
    if not generated or not gold:
        return False
    s = str(generated)
    pred = extract_answer(s) if "boxed" in s else s.strip()
    return bool(math_equal(pred, str(gold), timeout=True))


# ---------------------------------------------------------------------------
# Logprob extraction for \\boxed{} content
# ---------------------------------------------------------------------------

def extract_boxed_logprobs(
    generated_text: str,
    lp_tokens: List[Dict],
) -> Tuple[Optional[float], Optional[List[str]]]:
    """Sum logprobs for tokens that fall inside the first \\boxed{...}."""
    if not lp_tokens or not generated_text:
        return None, None

    m = re.search(r"\\boxed\{", generated_text)
    if not m:
        return None, None

    # Find matching closing brace (handles one level of nesting)
    depth, pos = 1, m.end()
    while pos < len(generated_text) and depth > 0:
        c = generated_text[pos]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None, None

    content_start, content_end = m.end(), pos - 1   # exclusive

    # Walk token byte positions to find overlap with boxed span
    char_pos = 0
    ans_lps: List[float] = []
    ans_toks: List[str] = []
    for lp in lp_tokens:
        raw_bytes = lp.get("bytes") or []
        byte_len = len(raw_bytes) if raw_bytes else len((lp.get("token") or "").encode("utf-8"))
        tok_end = char_pos + byte_len
        if char_pos >= content_start and tok_end <= content_end + 1:
            lp_val = lp.get("logprob")
            if lp_val is not None:
                ans_lps.append(float(lp_val))
                ans_toks.append(lp.get("token", ""))
        char_pos = tok_end

    if not ans_lps:
        return None, None
    return float(sum(ans_lps)), ans_toks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_step_prefix_eval(
    input_jsonl: str,
    model_id: str,
    api_base: str,
    output_jsonl: str,
    n_samples: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 1024,
    top_logprobs: int = 5,
):
    model = VLLMServerModel(
        model_id=model_id,
        api_base=api_base,
        api_key="EMPTY",
        logprobs=True,
        top_logprobs=top_logprobs,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(input_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} correct trajectories from {input_jsonl}")

    results = []
    for traj_idx, entry in enumerate(entries):
        question   = entry.get("question", "")
        true_answer = str(entry.get("true_answer", entry.get("answer", "")))
        log_data   = entry.get("log_data") or {}
        messages   = log_data.get("messages") or []

        if len(messages) < 5:
            print(f"[{traj_idx+1}] skipping — too few messages ({len(messages)})")
            continue

        task      = get_task_from_messages(messages)
        step_ends = find_step_end_indices(messages)
        n_steps   = len(step_ends)

        if n_steps == 0:
            print(f"[{traj_idx+1}] skipping — no step boundaries found")
            continue

        print(f"[{traj_idx+1}/{len(entries)}] n_steps={n_steps}  q={question[:60]!r}")

        step_results = []
        for step_i, end_idx in enumerate(step_ends):
            step_num  = step_i + 1
            fa_msgs   = build_final_answer_messages(messages[:end_idx], task)

            try:
                chat_msgs = model(fa_msgs, n=n_samples)
            except Exception as e:
                print(f"  step {step_num}: model call failed: {e}")
                step_results.append({
                    "step": step_num, "error": str(e),
                    "accuracy": None, "n_correct": None,
                    "n_total": n_samples, "generations": [],
                })
                continue

            if not isinstance(chat_msgs, list):
                chat_msgs = [chat_msgs]

            gens = []
            n_correct = 0
            for cm in chat_msgs:
                gen_text = cm.content or ""
                correct  = score_math_answer(gen_text, true_answer)
                if correct:
                    n_correct += 1

                lp_tokens        = _extract_generation_logprobs(cm)
                boxed_sum, b_toks = extract_boxed_logprobs(gen_text, lp_tokens or [])

                pred_str = str(gen_text)
                pred_ans = extract_answer(pred_str) if "boxed" in pred_str else pred_str.strip()

                gens.append({
                    "generated_text":    gen_text,
                    "extracted_answer":  pred_ans,
                    "is_correct":        correct,
                    "full_logprobs":     lp_tokens,
                    "answer_logprob_sum": boxed_sum,
                    "answer_tokens":     b_toks,
                })

            acc = n_correct / len(chat_msgs)
            print(f"  step {step_num}: acc={acc:.2f}  ({n_correct}/{len(chat_msgs)})")

            step_results.append({
                "step":          step_num,
                "prefix_length": end_idx,
                "n_correct":     n_correct,
                "n_total":       len(chat_msgs),
                "accuracy":      acc,
                "generations":   gens,
            })

        results.append({
            "question":     question,
            "true_answer":  true_answer,
            "n_steps":      n_steps,
            "step_results": step_results,
        })

    with open(output_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(results)} trajectory results to {output_jsonl}")

    # Aggregated per-step accuracy summary
    max_n = max((r["n_steps"] for r in results), default=0)
    print("\n=== Per-step accuracy summary ===")
    for s in range(1, max_n + 1):
        accs = [
            st["accuracy"]
            for r in results
            for st in r["step_results"]
            if st["step"] == s and st["accuracy"] is not None
        ]
        if accs:
            print(f"  step {s}: mean_acc={sum(accs)/len(accs):.3f}  n_traj={len(accs)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl",  required=True)
    ap.add_argument("--model_id",     default="Qwen/Qwen3-14B")
    ap.add_argument("--api_base",     default="http://127.0.0.1:8000/v1")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--n_samples",    type=int,   default=5)
    ap.add_argument("--temperature",  type=float, default=0.7)
    ap.add_argument("--top_p",        type=float, default=0.8)
    ap.add_argument("--max_tokens",   type=int,   default=1024)
    ap.add_argument("--top_logprobs", type=int,   default=5)
    args = ap.parse_args()
    run_step_prefix_eval(**vars(args))
