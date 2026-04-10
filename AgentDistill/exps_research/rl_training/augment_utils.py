"""
augment_utils.py — Question augmentation for Input Invariance Breaking (IIB) RL.

Produces surface-perturbed versions x' = aug(x) of a math question x while
preserving semantic content (same numbers, same problem structure).

Goal: teacher trajectories τ(x) and τ(x') should diverge (our training target)
while both still solve the problem correctly.

Strategy: template-based instruction-level paraphrase.
  - Changes the framing / phrasing of the question
  - Keeps all numbers, variables, and mathematical content intact
  - Requires no external LLM call
  - Cheap and deterministic given a template index
"""

from typing import List, Dict

# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------
# Each template wraps the raw question {q} in a different instructional frame.
# Template 0 is the identity (original phrasing) — used to verify round-trips.
# Templates 1-N are the augmentation variants.
#
# Design principle: vary the pragmatic framing (who is asking, what tone,
# what kind of answer is expected) so the model's "entry point" into reasoning
# differs, which tends to produce structurally different solution paths.

TEMPLATES: List[str] = [
    # 0 — identity (never used as aug; kept for indexing clarity)
    "{q}",

    # 1 — explicit step-by-step instruction
    "Solve the following problem step by step:\n{q}",

    # 2 — problem label prefix
    "Problem: {q}\n\nProvide a complete solution.",

    # 3 — student-poses framing
    "A student asks the following question:\n{q}\n\nGive a clear, correct answer.",

    # 4 — challenge framing
    "Mathematical challenge:\n{q}\n\nShow all your work and verify your answer.",

    # 5 — careful-analysis request
    "Carefully analyze and solve:\n{q}",

    # 6 — task-bracket framing
    "[Task]\n{q}\n[End Task]\n\nDemonstrate your solution approach.",

    # 7 — hint-appended framing
    "{q}\n\nHint: Consider breaking the problem into smaller steps before coding.",

    # 8 — formal-exam framing
    "Examination question:\n{q}\n\nPresent a rigorous, well-commented solution.",

    # 9 — teacher-to-student framing
    "Explain to a student how to solve:\n{q}\n\nInclude the final numerical answer.",
]

N_TEMPLATES = len(TEMPLATES) - 1  # number of augmentation variants (excluding identity)


def augment_question(question: str, template_idx: int) -> str:
    """
    Return the question wrapped in TEMPLATES[template_idx].

    Args:
        question:     Original question text.
        template_idx: Index in [1, N_TEMPLATES].  0 is identity and is skipped.
                      Values outside range are wrapped with modulo.

    Returns:
        Augmented question string.
    """
    # Clamp to [1, N_TEMPLATES] so we never return the identity template
    idx = 1 + ((template_idx - 1) % N_TEMPLATES)
    return TEMPLATES[idx].format(q=question)


def build_aug_examples(
    examples: List[dict],
    template_idx: int,
) -> tuple:
    """
    Given a list of question dicts (MATH-500 format), produce augmented versions.

    Args:
        examples:     List of dicts with at least "question" and "answer" fields.
        template_idx: Which template to use for augmentation.

    Returns:
        aug_examples:      List of dicts with augmented "question" field.
        aug_to_orig_map:   Dict mapping aug_question → orig_question.
    """
    aug_examples = []
    aug_to_orig_map: Dict[str, str] = {}

    for ex in examples:
        orig_q = ex["question"]
        aug_q = augment_question(orig_q, template_idx)
        aug_ex = {**ex, "question": aug_q}
        aug_examples.append(aug_ex)
        aug_to_orig_map[aug_q] = orig_q

    return aug_examples, aug_to_orig_map
