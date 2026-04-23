#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot assistant-only Code-token curves with a zoomed y-axis for low-probability patterns."
    )
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exclude_radius", type=int, default=6)
    parser.add_argument("--quantile", type=float, default=0.98)
    parser.add_argument("--min_upper", type=float, default=1e-4)
    parser.add_argument("--max_upper", type=float, default=0.2)
    return parser.parse_args()


def sanitize_plot_text(text: str) -> str:
    return str(text).replace("\\", r"\\").replace("$", r"\$")


def shorten(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def compute_zoom_upper(
    positions: Sequence[int],
    probs: Sequence[float],
    tool_call_starts: Sequence[int],
    exclude_radius: int,
    quantile: float,
    min_upper: float,
    max_upper: float,
) -> float:
    filtered: List[float] = []
    for pos, prob in zip(positions, probs):
        if any(abs(pos - marker) <= exclude_radius for marker in tool_call_starts):
            continue
        filtered.append(prob)
    if not filtered:
        filtered = list(probs)
    filtered = sorted(filtered)
    if not filtered:
        return max_upper
    idx = min(len(filtered) - 1, max(0, math.ceil(quantile * len(filtered)) - 1))
    upper = filtered[idx] * 1.1
    upper = max(upper, min_upper)
    upper = min(upper, max_upper)
    return upper


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.summary_json, "r") as f:
        rows = json.load(f)

    for row in rows:
        positions = row["curve_positions"]
        probs = row["curve_probs"]
        assistant_spans = row["assistant_token_spans"]
        tool_call_starts = row["actual_tool_call_starts"]
        full_end = max((span[1] for span in assistant_spans), default=max(positions) if positions else 0)
        full_axis_positions = list(range(1, full_end + 1))
        prob_by_position = {pos: prob for pos, prob in zip(positions, probs)}
        full_axis_probs = [prob_by_position.get(pos, float("nan")) for pos in full_axis_positions]

        upper = compute_zoom_upper(
            positions,
            probs,
            tool_call_starts,
            exclude_radius=args.exclude_radius,
            quantile=args.quantile,
            min_upper=args.min_upper,
            max_upper=args.max_upper,
        )

        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.plot(full_axis_positions, full_axis_probs, color="steelblue", linewidth=1.5, label="P(next token = 'Code')")
        for idx, marker in enumerate(tool_call_starts):
            ax.axvline(
                marker,
                color="crimson",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label="actual tool-call start" if idx == 0 else None,
            )
        ax.set_ylim(0.0, upper)
        ax.set_xlabel("Full-token position across the cleaned full sequence")
        ax.set_ylabel("Probability (zoomed)")
        ax.set_title(
            f"sample={row['raw_index']} steps={row['step_count']} "
            f"correct={'yes' if row['correct'] > 0 else 'no'} zoom_upper={upper:.4g}"
        )
        question_preview = sanitize_plot_text(shorten(row.get("question", "")))
        answer_preview = sanitize_plot_text(shorten(row.get("generated_answer", ""), limit=90))
        text_box = (
            f"Question: {question_preview}\n"
            f"Answer: {answer_preview}\n"
            f"Tool-call starts: {tool_call_starts}"
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
        out_path = output_dir / f"trajectory_{row['local_index']:02d}_sample_{row['raw_index']:04d}_zoomed.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"saved={out_path} zoom_upper={upper:.6g}")


if __name__ == "__main__":
    main()
