"""
augment_utils.py — Question augmentation for Input Invariance Breaking (IIB) RL.

Produces surface-perturbed versions x' = aug(x) of a math question x while
preserving semantic equivalence (same problem, same answer).

Three-level augmentation (applied in order):
  Level 1 — Instruction template  (deterministic, recipe-based)
  Level 2 — Content transform     (deterministic, recipe-based)
  Level 3 — Random noise          (stochastic, seeded per question)

Noise reproducibility:
    Each question gets its own RNG seeded by hash(question) ^ recipe_idx,
    so the same question always receives the same noise, but different questions
    receive different perturbations within the same batch.

No external LLM call required.  All transforms are regex-based.
"""

import re
import random
from typing import List, Tuple, Callable, Optional

# ---------------------------------------------------------------------------
# Level-1: Instruction templates
# ---------------------------------------------------------------------------

_TEMPLATES = [
    "{q}",                                                           # 0: identity
    "Solve the following problem step by step:\n{q}",                # 1
    "Problem:\n{q}\n\nProvide a complete solution.",                  # 2
    "A student poses this question:\n{q}\n\nGive a correct answer.",  # 3
    "Mathematical challenge:\n{q}\n\nShow all work.",                 # 4
]

def _apply_template(q: str, idx: int) -> str:
    idx = 1 + ((idx - 1) % (len(_TEMPLATES) - 1))
    return _TEMPLATES[idx].format(q=q)


# ---------------------------------------------------------------------------
# Level-2: Deterministic content transforms
# ---------------------------------------------------------------------------

# ---- synonym replacement (deterministic: first match) ----
_VERB_SYNONYMS: List[Tuple[str, str]] = [
    (r'\bfind\b',      'calculate'),
    (r'\bcalculate\b', 'determine'),
    (r'\bdetermine\b', 'compute'),
    (r'\bcompute\b',   'find'),
    (r'\bevaluate\b',  'calculate'),
    (r'\bsolve\b',     'work out'),
    (r'\bwork out\b',  'solve'),
]

def _synonym_replace(text: str) -> str:
    for pattern, replacement in _VERB_SYNONYMS:
        new = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if new != text:
            return new
    return text


# ---- fraction ↔ decimal ----
_SIMPLE_FRACS = {
    '1/2':'0.5',   '1/3':'0.333',  '2/3':'0.667',
    '1/4':'0.25',  '3/4':'0.75',
    '1/5':'0.2',   '2/5':'0.4',    '3/5':'0.6',   '4/5':'0.8',
    '1/8':'0.125', '3/8':'0.375',  '5/8':'0.625', '7/8':'0.875',
    '1/10':'0.1',  '3/10':'0.3',   '7/10':'0.7',  '9/10':'0.9',
}
_DEC_TO_FRAC = {v: k for k, v in _SIMPLE_FRACS.items()}

def _frac_to_decimal(text: str) -> str:
    for frac, dec in _SIMPLE_FRACS.items():
        text = text.replace(frac, dec)
    return text

def _decimal_to_frac(text: str) -> str:
    for dec, frac in _DEC_TO_FRAC.items():
        text = re.sub(r'(?<!\d)' + re.escape(dec) + r'(?!\d)', frac, text)
    return text


# ---- percentage ↔ decimal ----
def _pct_to_decimal(text: str) -> str:
    def _conv(m):
        pct = float(m.group(1))
        return str(round(pct / 100, 6)).rstrip('0').rstrip('.')
    return re.sub(r'(\d+(?:\.\d+)?)\s*%', _conv, text)

def _decimal_to_pct(text: str) -> str:
    def _conv(m):
        val = float(m.group(0))
        if 0 < val < 1:
            pct = round(val * 100, 4)
            fmt = int(pct) if pct == int(pct) else pct
            return f"{fmt}%"
        return m.group(0)
    return re.sub(r'0\.\d+', _conv, text)


# ---- small integer ↔ word ----
_INT_TO_WORD = {
    '1':'one','2':'two','3':'three','4':'four','5':'five',
    '6':'six','7':'seven','8':'eight','9':'nine','10':'ten',
    '11':'eleven','12':'twelve',
}
_WORD_TO_INT = {v: k for k, v in _INT_TO_WORD.items()}

def _int_to_word(text: str) -> str:
    def _rep(m):
        return _INT_TO_WORD.get(m.group(0), m.group(0))
    return re.sub(r'(?<![.\d])([1-9]|1[0-2])(?![.\d%/])', _rep, text, count=1)

def _word_to_int(text: str) -> str:
    pat = r'\b(' + '|'.join(re.escape(w) for w in _WORD_TO_INT) + r')\b'
    return re.sub(pat, lambda m: _WORD_TO_INT[m.group(0).lower()],
                  text, count=1, flags=re.IGNORECASE)


# ---- condition reorder ----
def _reorder_conditions(text: str) -> str:
    kv = re.compile(r'(\b\w+\s*=\s*[\w.]+)(\s*,\s*\b\w+\s*=\s*[\w.]+)+')
    m = kv.search(text)
    if m:
        parts = [p.strip() for p in m.group(0).split(',')]
        if len(parts) >= 2:
            return text[:m.start()] + ', '.join(parts[::-1]) + text[m.end():]
    sentences = re.split(r'(?<=[.!?])\s+', text, maxsplit=2)
    if len(sentences) >= 2:
        sentences[0], sentences[1] = sentences[1], sentences[0]
        return ' '.join(sentences)
    return text


# ---------------------------------------------------------------------------
# Level-3: Random noise transforms
# ---------------------------------------------------------------------------
# Each noise function takes (text: str, rng: random.Random) → str.
# They are designed to be lightweight and always semantically neutral.

# ---- 3a: trailing zeros on integers ----
# "has 5 apples" → "has 5.0 apples"  (model may parse differently as float)
def _noise_trailing_zero(text: str, rng: random.Random) -> str:
    """Add '.0' to one randomly chosen standalone integer (not already decimal)."""
    matches = list(re.finditer(r'(?<![.\d/])(\d+)(?![.\d/%])', text))
    # Exclude very large numbers (>4 digits) — they're likely years / IDs
    matches = [m for m in matches if len(m.group(1)) <= 4]
    if not matches:
        return text
    target = rng.choice(matches)
    return text[:target.end()] + '.0' + text[target.end():]


# ---- 3b: random synonym selection from an expanded set ----
_VERB_SYN_EXPANDED = {
    r'\bfind\b':      ['calculate', 'determine', 'compute', 'obtain', 'derive'],
    r'\bcalculate\b': ['determine', 'compute', 'find', 'work out', 'evaluate'],
    r'\bdetermine\b': ['find', 'compute', 'calculate', 'establish', 'identify'],
    r'\bsolve\b':     ['work out', 'figure out', 'resolve', 'calculate'],
    r'\bevaluate\b':  ['calculate', 'compute', 'assess', 'determine'],
    r'\bcompute\b':   ['calculate', 'determine', 'find', 'evaluate'],
}

def _noise_random_synonym(text: str, rng: random.Random) -> str:
    """Replace one math-verb with a randomly chosen synonym."""
    items = list(_VERB_SYN_EXPANDED.items())
    rng.shuffle(items)
    for pattern, choices in items:
        if re.search(pattern, text, flags=re.IGNORECASE):
            replacement = rng.choice(choices)
            return re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
    return text


# ---- 3c: filler phrase insertion ----
_FILLERS = [
    "Note that ",
    "Recall that ",
    "We know that ",
    "Given that ",
    "Observe that ",
    "It is known that ",
]

def _noise_filler(text: str, rng: random.Random) -> str:
    """Prepend a random filler phrase to the first sentence of the question."""
    # Only insert if the text starts with an uppercase letter (looks like a sentence)
    if not text or not text[0].isupper():
        return text
    filler = rng.choice(_FILLERS)
    # Lower-case the first char of the original text after the filler
    return filler + text[0].lower() + text[1:]


# ---- 3d: number elaboration (digit + parenthetical word) ----
# "3 items" → "3 (three) items"
def _noise_number_elaboration(text: str, rng: random.Random) -> str:
    """Add a parenthetical word form next to one small integer."""
    matches = list(re.finditer(r'(?<![.\d])([1-9]|1[0-2])(?![.\d%/])', text))
    matches = [m for m in matches if m.group(0) in _INT_TO_WORD]
    if not matches:
        return text
    target = rng.choice(matches)
    word = _INT_TO_WORD[target.group(0)]
    return text[:target.end()] + f" ({word})" + text[target.end():]


# ---- 3e: punctuation / whitespace variation ----
def _noise_punctuation(text: str, rng: random.Random) -> str:
    """Randomly add or remove a trailing period, or add a blank line mid-text."""
    choice = rng.randint(0, 2)
    if choice == 0:
        # Toggle trailing period
        if text.endswith('.'):
            return text[:-1]
        else:
            return text + '.'
    elif choice == 1:
        # Add double space after the first period (whitespace variant)
        return re.sub(r'\.\s+', '.  ', text, count=1)
    else:
        # Add trailing newline (blank line) — some models are sensitive to this
        return text + '\n'


# ---- 3f: explicit multiplication form ----
# "20% of 50" → "0.20 × 50"  (changes the surface computation)
def _noise_pct_of(text: str, rng: random.Random) -> str:
    """Convert 'N% of M' to '(N/100)*M' or 'N/100 of M'."""
    def _conv(m):
        pct_str = m.group(1)
        of_str  = m.group(2)
        choice = rng.randint(0, 1)
        if choice == 0:
            return f"({pct_str}/100) * {of_str}"
        else:
            pct_val = float(pct_str)
            decimal = str(round(pct_val / 100, 6)).rstrip('0').rstrip('.')
            return f"{decimal} * {of_str}"
    return re.sub(r'(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)', _conv, text)


# Pool of noise functions to sample from
_NOISE_POOL: List[Callable] = [
    _noise_trailing_zero,
    _noise_random_synonym,
    _noise_filler,
    _noise_number_elaboration,
    _noise_punctuation,
    _noise_pct_of,
]


def _apply_noise(text: str, rng: random.Random, n_ops: int = 2) -> str:
    """
    Apply n_ops randomly chosen noise operations from _NOISE_POOL.

    Operations are sampled without replacement so no op runs twice.
    n_ops is capped at len(_NOISE_POOL).
    """
    ops = rng.sample(_NOISE_POOL, k=min(n_ops, len(_NOISE_POOL)))
    for fn in ops:
        text = fn(text, rng)
    return text


# ---------------------------------------------------------------------------
# Augmentation recipes  (deterministic part)
# ---------------------------------------------------------------------------
# Each recipe = (template_idx, [deterministic_transforms]).
# Recipe 0 = identity.

Recipe = List[Callable[[str], str]]

_RECIPES: List[Tuple[int, Recipe]] = [
    (0, []),                                          # 0: identity
    (1, [_frac_to_decimal]),                          # 1
    (2, [_pct_to_decimal]),                           # 2
    (3, [_synonym_replace]),                          # 3
    (1, [_int_to_word]),                              # 4
    (2, [_reorder_conditions]),                       # 5
    (3, [_frac_to_decimal, _synonym_replace]),        # 6
    (4, [_pct_to_decimal, _int_to_word]),             # 7
    (1, [_decimal_to_frac]),                          # 8
    (2, [_decimal_to_pct, _synonym_replace]),         # 9
    (3, [_word_to_int, _reorder_conditions]),         # 10
    (4, [_frac_to_decimal, _reorder_conditions]),     # 11
]

N_RECIPES = len(_RECIPES) - 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def augment_question(
    question: str,
    recipe_idx: int,
    noise_ops: int = 2,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Augment a question with deterministic recipe + random noise.

    Args:
        question:   Original question text.
        recipe_idx: Which deterministic recipe to apply (1..N_RECIPES; wraps).
        noise_ops:  Number of random noise operations to apply (0 = noise off).
        rng:        Seeded Random instance.  If None, seeded from question hash
                    (reproducible per question, varies across questions).

    Returns:
        Augmented question string (semantically equivalent to original).
    """
    if rng is None:
        seed = hash(question) ^ recipe_idx
        rng = random.Random(seed & 0xFFFFFFFF)

    # Level 2: deterministic content transform
    idx = 1 + ((recipe_idx - 1) % N_RECIPES)
    tmpl_idx, transforms = _RECIPES[idx]
    text = question
    for fn in transforms:
        text = fn(text)

    # Level 3: random noise
    if noise_ops > 0:
        text = _apply_noise(text, rng, n_ops=noise_ops)

    # Level 1: instruction template
    if tmpl_idx > 0:
        text = _apply_template(text, tmpl_idx)

    return text


def build_aug_examples(
    examples: List[dict],
    recipe_idx: int,
    noise_ops: int = 2,
) -> Tuple[List[dict], dict]:
    """
    Produce augmented versions of a list of question dicts.

    Each question gets its own seeded RNG (hash(question) ^ recipe_idx),
    so noise is reproducible per question but varies across questions.

    Args:
        examples:    List of dicts with "question" and "answer" fields.
        recipe_idx:  Which recipe to use.
        noise_ops:   Number of random noise ops per question (0 = noise off).

    Returns:
        aug_examples:  List of dicts with augmented "question" field.
        aug_to_orig:   Dict mapping aug_question → orig_question.
    """
    aug_examples = []
    aug_to_orig: dict = {}

    for ex in examples:
        orig_q = ex["question"]
        rng = random.Random(hash(orig_q) ^ recipe_idx & 0xFFFFFFFF)
        aug_q = augment_question(orig_q, recipe_idx, noise_ops=noise_ops, rng=rng)
        aug_ex = {**ex, "question": aug_q}
        aug_examples.append(aug_ex)
        aug_to_orig[aug_q] = orig_q

    return aug_examples, aug_to_orig
