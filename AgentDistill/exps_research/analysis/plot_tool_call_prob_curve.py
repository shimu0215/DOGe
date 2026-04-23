#!/usr/bin/env python3
import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exps_research.train_utils.preprocess import PROMPT_TEMPLATES, clean_messages
from exps_research.unified_framework.score_answers import evaluate_math_answer
from smolagents import FinalAnswerTool
from smolagents.agents import populate_template


@dataclass
class MarkerSpan:
    message_idx: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    token_ids: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the probability of the tool-call start marker across full smolagents trajectories."
    )
    parser.add_argument("--trajectory_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--marker_text", type=str, default="Code:")
    parser.add_argument("--probe_text", type=str, default="Code:")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def build_python_only_system_prompt() -> str:
    prompt_template = PROMPT_TEMPLATES["system_prompt_short"]
    tools = {"final_answer": FinalAnswerTool()}
    return populate_template(prompt_template, variables={"tools": tools})


def load_rows(path: str, limit: int) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r") as f:
        for raw_idx, line in enumerate(f):
            row = json.loads(line)
            row["_raw_index"] = raw_idx
            if row.get("log_data") is None:
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def prepare_messages(row: Dict, system_prompt: str) -> List[Dict[str, str]]:
    messages = clean_messages(copy.deepcopy(row["log_data"]["messages"]))
    if messages:
        messages[0]["content"] = system_prompt
    return messages


def tokenize_messages(tokenizer, messages: Sequence[Dict[str, str]]) -> List[int]:
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=False,
    )


def align_prefix(full_ids: List[int], prefix_ids: List[int]) -> int:
    if len(prefix_ids) <= len(full_ids) and full_ids[: len(prefix_ids)] == prefix_ids:
        return len(prefix_ids)
    trimmed = list(prefix_ids)
    while trimmed and len(trimmed) <= len(full_ids):
        trimmed = trimmed[:-1]
        if full_ids[: len(trimmed)] == trimmed:
            return len(trimmed)
    raise ValueError("Unable to align prefix tokens with the full trajectory tokens.")


def make_truncated_messages(
    messages: Sequence[Dict[str, str]],
    message_idx: int,
    char_limit: int,
) -> List[Dict[str, str]]:
    truncated = copy.deepcopy(list(messages[: message_idx + 1]))
    truncated[-1]["content"] = truncated[-1]["content"][:char_limit]
    return truncated


def locate_marker_spans(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    marker_text: str,
) -> tuple[List[int], List[MarkerSpan]]:
    full_ids = tokenize_messages(tokenizer, messages)
    spans: List[MarkerSpan] = []
    for message_idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", ""))
        search_from = 0
        while True:
            char_start = content.find(marker_text, search_from)
            if char_start < 0:
                break
            char_end = char_start + len(marker_text)
            before_ids = tokenize_messages(
                tokenizer,
                make_truncated_messages(messages, message_idx, char_start),
            )
            through_ids = tokenize_messages(
                tokenizer,
                make_truncated_messages(messages, message_idx, char_end),
            )
            token_start = align_prefix(full_ids, before_ids)
            token_end = align_prefix(full_ids, through_ids)
            spans.append(
                MarkerSpan(
                    message_idx=message_idx,
                    char_start=char_start,
                    char_end=char_end,
                    token_start=token_start,
                    token_end=token_end,
                    token_ids=full_ids[token_start:token_end],
                )
            )
            search_from = char_end
    return full_ids, spans


def get_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def compute_probe_curve(
    model,
    input_ids: List[int],
    probe_token_id: int,
) -> List[float]:
    input_device = next(model.parameters()).device
    inputs = torch.tensor([input_ids], dtype=torch.long, device=input_device)
    with torch.no_grad():
        outputs = model(input_ids=inputs, use_cache=False)
    logits = outputs.logits[0, :-1].float()
    selected = logits[:, probe_token_id]
    normalization = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(selected - normalization)
    return probs.cpu().tolist()


def compute_correctness(row: Dict) -> float:
    result = evaluate_math_answer(
        model=None,
        predicted=row.get("generated_answer"),
        gold=row.get("true_answer"),
        question=row.get("question"),
        do_extract_answer=False,
    )
    return float(result["score"])


def count_action_steps(row: Dict) -> int:
    traces = row.get("log_data", {}).get("generation_trace", [])
    return sum(1 for trace in traces if trace.get("step_type") in {"action", "action_finalize"})


def shorten(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def plot_one(
    output_path: Path,
    curve: Sequence[float],
    spans: Sequence[MarkerSpan],
    row: Dict,
    step_count: int,
    correctness: float,
    probe_text: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x_positions = list(range(1, len(curve) + 1))
    ax.plot(x_positions, curve, color="steelblue", linewidth=1.5, label=f"P(next token starts {probe_text!r})")
    for idx, span in enumerate(spans):
        ax.axvline(
            span.token_start,
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="actual tool-call start" if idx == 0 else None,
        )
    ax.set_xlabel("Token position in full cleaned trajectory")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"sample={row['_raw_index']} steps={step_count} "
        f"correct={'yes' if correctness > 0 else 'no'} tool_calls={len(spans)}"
    )
    question_preview = shorten(row.get("question", ""))
    answer_preview = shorten(row.get("generated_answer", ""), limit=90)
    text_box = (
        f"Question: {question_preview}\n"
        f"Answer: {answer_preview}\n"
        f"Marker token starts: {[span.token_start for span in spans]}"
    )
    ax.text(
        0.01,
        0.98,
        text_box,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = build_python_only_system_prompt()
    rows = load_rows(args.trajectory_path, args.limit)
    if not rows:
        raise RuntimeError("No usable rows with log_data were found.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    probe_token_ids = tokenizer.encode(args.probe_text, add_special_tokens=False)
    if not probe_token_ids:
        raise ValueError(f"Probe text {args.probe_text!r} produced no tokens.")
    probe_token_id = probe_token_ids[0]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=get_torch_dtype(args.torch_dtype),
        device_map="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    summary_rows = []
    for local_idx, row in enumerate(rows):
        messages = prepare_messages(row, system_prompt)
        input_ids, spans = locate_marker_spans(tokenizer, messages, args.marker_text)
        curve = compute_probe_curve(model, input_ids, probe_token_id)
        step_count = count_action_steps(row)
        correctness = compute_correctness(row)

        figure_path = output_dir / f"trajectory_{local_idx:02d}_sample_{row['_raw_index']:04d}.png"
        plot_one(
            output_path=figure_path,
            curve=curve,
            spans=spans,
            row=row,
            step_count=step_count,
            correctness=correctness,
            probe_text=args.probe_text,
        )

        summary_rows.append(
            {
                "local_index": local_idx,
                "raw_index": row["_raw_index"],
                "question": row.get("question", ""),
                "true_answer": row.get("true_answer", ""),
                "generated_answer": row.get("generated_answer", ""),
                "correct": correctness,
                "step_count": step_count,
                "tool_call_count": len(spans),
                "tool_call_token_starts": [span.token_start for span in spans],
                "tool_call_token_ends": [span.token_end for span in spans],
                "tool_call_marker_token_ids": [span.token_ids for span in spans],
                "probe_text": args.probe_text,
                "probe_token_ids": probe_token_ids,
                "figure_path": str(figure_path),
            }
        )
        print(
            f"[{local_idx + 1}/{len(rows)}] sample={row['_raw_index']} "
            f"steps={step_count} correct={correctness:.0f} tool_calls={len(spans)} "
            f"saved={figure_path}"
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
