"""
augment_utils.py — Question augmentation for Input Invariance Breaking (IIB) RL.

Produces surface-perturbed versions x' = aug(x) of a math question x while
preserving semantic equivalence (same problem, same answer).

Goal: teacher trajectories τ(x) and τ(x') should diverge because the model
processes the problem differently at the representation level — not just because
of a changed instruction prefix.

Two-level augmentation:
  Level 1 — Instruction template:
      Changes the pragmatic framing (who is asking, what tone).
      Weak effect on its own; used as a wrapper around Level 2.

  Level 2 — Content transform:
      Changes the mathematical representation of numbers/units/phrasing so
      the model's internal parsing differs and it tends to write different code.

      Transforms (all reversible / semantically equivalent):
        frac→decimal   "3/4"          → "0.75"
        decimal→frac   "0.75"         → "3/4"
        pct→decimal    "75%"          → "0.75"
        decimal→pct    "0.75"         → "75%"
        int→word       "2 cats"       → "two cats"   (small integers ≤ 12)
        word→int       "two cats"     → "2 cats"
        synonym        "find"         → "calculate" / "determine" / "compute"
        reorder-conds  "A=3, B=5 ..." → "B=5, A=3 ..." (swap sentence pairs)

Each augmentation "recipe" is a (template_idx, [transform, ...]) pair.
Recipes are indexed 1..N and rotated across resample cycles.

No external LLM call required.  All transforms are regex-based.
"""

import re
import random
from typing import List, Tuple, Callable

# ---------------------------------------------------------------------------
# Level-1: Instruction templates
# ---------------------------------------------------------------------------
# Template 0 = identity.  Templates 1-4 are the framing variants.

_TEMPLATES = [
    "{q}",                                                        # 0: identity
    "Solve the following problem step by step:\n{q}",             # 1
    "Problem:\n{q}\n\nProvide a complete solution.",              # 2
    "A student poses this question:\n{q}\n\nGive a correct answer.",  # 3
    "Mathematical challenge:\n{q}\n\nShow all work.",             # 4
]

def _apply_template(q: str, idx: int) -> str:
    idx = 1 + ((idx - 1) % (len(_TEMPLATES) - 1))
    return _TEMPLATES[idx].format(q=q)


# ---------------------------------------------------------------------------
# Level-2: Content transforms
# ---------------------------------------------------------------------------

# ---- synonym replacement ----
_VERB_SYNONYMS: List[Tuple[str, str]] = [
    (r'\bfind\b',       'calculate'),
    (r'\bcalculate\b',  'determine'),
    (r'\bdetermine\b',  'compute'),
    (r'\bcompute\b',    'find'),
    (r'\bevaluate\b',   'calculate'),
    (r'\bsolve\b',      'work out'),
    (r'\bwork out\b',   'solve'),
]

def _synonym_replace(text: str) -> str:
    """Replace one math-verb with a synonym (first match wins)."""
    for pattern, replacement in _VERB_SYNONYMS:
        new = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if new != text:
            return new
    return text


# ---- fraction ↔ decimal ----
_SIMPLE_FRACS = {
    '1/2': '0.5',  '1/3': '0.333',  '2/3': '0.667',
    '1/4': '0.25', '3/4': '0.75',
    '1/5': '0.2',  '2/5': '0.4',    '3/5': '0.6',   '4/5': '0.8',
    '1/8': '0.125','3/8': '0.375',  '5/8': '0.625', '7/8': '0.875',
    '1/10':'0.1',  '3/10':'0.3',    '7/10':'0.7',   '9/10':'0.9',
}
_DEC_TO_FRAC = {v: k for k, v in _SIMPLE_FRACS.items()}

def _frac_to_decimal(text: str) -> str:
    for frac, dec in _SIMPLE_FRACS.items():
        text = text.replace(frac, dec)
    return text

def _decimal_to_frac(text: str) -> str:
    for dec, frac in _DEC_TO_FRAC.items():
        # Only replace standalone decimals (word-boundary on the right)
        text = re.sub(r'(?<!\d)' + re.escape(dec) + r'(?!\d)', frac, text)
    return text


# ---- percentage ↔ decimal ----
def _pct_to_decimal(text: str) -> str:
    """Convert 'N%' → '0.N' for integer percentages."""
    def _convert(m):
        pct = float(m.group(1))
        return str(round(pct / 100, 6)).rstrip('0').rstrip('.')
    return re.sub(r'(\d+(?:\.\d+)?)\s*%', _convert, text)

def _decimal_to_pct(text: str) -> str:
    """Convert standalone decimals like 0.75 → 75% (only obvious cases)."""
    def _convert(m):
        val = float(m.group(0))
        if 0 < val < 1:
            pct = round(val * 100, 4)
            fmt = int(pct) if pct == int(pct) else pct
            return f"{fmt}%"
        return m.group(0)
    return re.sub(r'0\.\d+', _convert, text)


# ---- small integer ↔ word ----
_INT_TO_WORD = {
    '1':'one','2':'two','3':'three','4':'four','5':'five',
    '6':'six','7':'seven','8':'eight','9':'nine','10':'ten',
    '11':'eleven','12':'twelve',
}
_WORD_TO_INT = {v: k for k, v in _INT_TO_WORD.items()}

def _int_to_word(text: str) -> str:
    """Replace a small standalone integer with its word form (first occurrence)."""
    def _replace(m):
        d = m.group(0)
        return _INT_TO_WORD.get(d, d)
    # Only replace integers NOT adjacent to other digits / decimal points
    return re.sub(r'(?<![.\d])([1-9]|1[0-2])(?![.\d%/])', _replace, text, count=1)

def _word_to_int(text: str) -> str:
    """Replace a number word with its digit form (first occurrence)."""
    pattern = r'\b(' + '|'.join(re.escape(w) for w in _WORD_TO_INT) + r')\b'
    def _replace(m):
        return _WORD_TO_INT[m.group(0).lower()]
    return re.sub(pattern, _replace, text, count=1, flags=re.IGNORECASE)


# ---- condition reorder ----
def _reorder_conditions(text: str) -> str:
    """
    If the question contains comma-separated 'key=value' pairs (e.g.,
    'a = 3, b = 5'), reverse their order.  Falls back to swapping the
    first two sentences if no such pattern is found.
    """
    # Try key=value pairs: "a = 3, b = 5, ..."
    kv_pattern = re.compile(
        r'(\b\w+\s*=\s*[\w.]+)(\s*,\s*\b\w+\s*=\s*[\w.]+)+'
    )
    m = kv_pattern.search(text)
    if m:
        span = m.group(0)
        parts = [p.strip() for p in span.split(',')]
        if len(parts) >= 2:
            parts_rev = parts[::-1]
            return text[:m.start()] + ', '.join(parts_rev) + text[m.end():]

    # Fallback: swap first two sentences
    sentences = re.split(r'(?<=[.!?])\s+', text, maxsplit=2)
    if len(sentences) >= 2:
        sentences[0], sentences[1] = sentences[1], sentences[0]
        return ' '.join(sentences)
    return text


# ---------------------------------------------------------------------------
# Augmentation recipes
# ---------------------------------------------------------------------------
# Each recipe is a list of content transforms applied in sequence,
# then wrapped in an instruction template.
# Recipe index 0 is identity (not used for augmentation).

Recipe = List[Callable[[str], str]]

_RECIPES: List[Tuple[int, Recipe]] = [
    # (template_idx, [transform, ...])
    (0, []),                                          # 0: identity
    (1, [_frac_to_decimal]),                          # 1: fractions → decimals
    (2, [_pct_to_decimal]),                           # 2: percentages → decimals
    (3, [_synonym_replace]),                          # 3: verb synonym
    (1, [_int_to_word]),                              # 4: digits → words
    (2, [_reorder_conditions]),                       # 5: reorder conditions
    (3, [_frac_to_decimal, _synonym_replace]),        # 6: fracs + synonym
    (4, [_pct_to_decimal, _int_to_word]),             # 7: pct + word
    (1, [_decimal_to_frac]),                          # 8: decimals → fractions
    (2, [_decimal_to_pct, _synonym_replace]),         # 9: decimal → pct + synonym
    (3, [_word_to_int, _reorder_conditions]),         # 10: word → int + reorder
    (4, [_frac_to_decimal, _reorder_conditions]),     # 11: fracs + reorder
]

N_RECIPES = len(_RECIPES) - 1   # number of augmentation variants (excl. identity)


def augment_question(question: str, recipe_idx: int) -> str:
    """
    Apply augmentation recipe `recipe_idx` to `question`.

    Args:
        question:   Original question text.
        recipe_idx: Index in [1, N_RECIPES].  0 is identity and is never used
                    for augmentation.  Values outside range wrap with modulo.

    Returns:
        Augmented question string.
    """
    idx = 1 + ((recipe_idx - 1) % N_RECIPES)
    tmpl_idx, transforms = _RECIPES[idx]

    text = question
    for fn in transforms:
        text = fn(text)

    if tmpl_idx > 0:
        text = _apply_template(text, tmpl_idx)

    return text


def build_aug_examples(
    examples: List[dict],
    recipe_idx: int,
) -> Tuple[List[dict], dict]:
    """
    Given a list of question dicts (MATH-500 format), produce augmented versions.

    Args:
        examples:    List of dicts with at least "question" and "answer" fields.
        recipe_idx:  Which recipe to use for augmentation.

    Returns:
        aug_examples:   List of dicts with augmented "question" field.
        aug_to_orig:    Dict mapping aug_question → orig_question.
    """
    aug_examples = []
    aug_to_orig: dict = {}

    for ex in examples:
        orig_q = ex["question"]
        aug_q = augment_question(orig_q, recipe_idx)
        aug_ex = {**ex, "question": aug_q}
        aug_examples.append(aug_ex)
        aug_to_orig[aug_q] = orig_q

    return aug_examples, aug_to_orig
