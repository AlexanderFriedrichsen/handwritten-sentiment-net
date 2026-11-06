"""Label-free sentence encoder.

Implements the mechanical steps fixed in authoring/RATIONALE.md (the
"Network architecture" entry's steps 1-7, and the "Composition/fusion/
output design" entry's pooling-formula amendment to step 6). This
module makes no sentiment judgment of its own: every value it reads
comes from the embedding table E, which scripts/compile_weights.py
compiles from the hand-authored authoring/*.yaml files. encode_sentence
never runs the composition/fusion/output layers, never sees a label, and
its control flow depends only on closed-class structural columns
(is_negator, is_contrast, degree_modifier) -- themselves data, not a
per-word special case in this code.

Steps:
  1. tokenize (handnet.tokenize.tokenize)
  2. vocab lookup -> token indices (UNK=0 for anything unseen)
  3. clause split on the first is_contrast token
  4. negation-scope window (SCOPE_WINDOW tokens, resets early at strong
     punctuation or at any is_contrast token)
  5. per-clause bucket assignment: plain / negated / modified (negated
     takes precedence over modified; negators and contrast markers are
     excluded from every bucket)
  6. pool each of the 6 clause x bucket buckets into a 7-dim vector by
     SUM then divide by sentence_length (not bucket size), with the
     negated bucket's negation_sensitivity column and the modified
     bucket's degree_modifier column redefined to a per-token product
     computed before pooling
  7. emit the 6 pooled 7-dim vectors (42 numbers) plus 7 sentence-level
     scalars = 49 numbers, in the index layout fixed by handnet.units
"""

from __future__ import annotations

import numpy as np

from handnet.columns import (
    CERTAINTY,
    CONTENT_NEGATOR,
    DEGREE_MODIFIER,
    IS_CONTRAST,
    IS_NEGATOR,
    NEGATION_SENSITIVITY,
    N_EMBED_COLS,
    VALENCE,
)
from handnet.tokenize import tokenize
from handnet.units import BLOCK_BASE, N_ENCODER_OUTPUTS, SCALAR_INDEX

SCOPE_WINDOW = 4
STRONG_PUNCT = frozenset({".", "!", "?", ";", ":"})

_EMPTY_BUCKETS = {"plain": (), "negated": (), "modified": ()}


def _token_indices(tokens: list[str], word_to_index: dict) -> list[int]:
    return [word_to_index.get(tok, 0) for tok in tokens]


def _negation_scope(tokens: list[str], indices: list[int], E: np.ndarray) -> list[bool]:
    """True at position t iff t falls within an earlier negator's
    forward window of SCOPE_WINDOW tokens (every token counts, content
    or function). Resets early (closes the window) at strong
    punctuation or at any is_contrast token -- a discourse pivot word
    always closes an open window, even one not chosen as the clause
    split point (e.g. a second "but" later in the sentence).
    """
    in_scope = [False] * len(tokens)
    remaining = 0
    for t, (tok, idx) in enumerate(zip(tokens, indices)):
        if tok in STRONG_PUNCT or E[idx, IS_CONTRAST] == 1.0:
            remaining = 0
            continue
        if E[idx, IS_NEGATOR] == 1.0:
            remaining = SCOPE_WINDOW
            continue
        in_scope[t] = remaining > 0
        if remaining > 0:
            remaining -= 1
    return in_scope


def _find_split(indices: list[int], E: np.ndarray) -> int | None:
    for t, idx in enumerate(indices):
        if E[idx, IS_CONTRAST] == 1.0:
            return t
    return None


def _assign_buckets(clause_indices: list[int], clause_scope: list[bool], E: np.ndarray) -> dict:
    """Returns {"plain": [...], "negated": [...], "modified": [...]},
    each a list of (token_index, prev_token_index_or_None) pairs.
    prev_token_index is the immediately preceding token *within this
    clause* (None for the clause's first token) and is only used by the
    "modified" bucket's redefined pooling formula.
    """
    buckets: dict = {"plain": [], "negated": [], "modified": []}
    for t, idx in enumerate(clause_indices):
        if E[idx, IS_NEGATOR] == 1.0 or E[idx, IS_CONTRAST] == 1.0:
            continue  # structurally excluded from every bucket
        if clause_scope[t]:
            buckets["negated"].append((idx, None))
            continue
        prev_idx = clause_indices[t - 1] if t > 0 else None
        if prev_idx is not None and E[prev_idx, DEGREE_MODIFIER] != 0.0:
            buckets["modified"].append((idx, prev_idx))
        else:
            buckets["plain"].append((idx, None))
    return buckets


def _pool_block(
    members, E: np.ndarray, sentence_length: int, redefine_negated: bool, redefine_modified: bool
) -> np.ndarray:
    """Pools `members` (token_index, prev_token_index_or_None pairs)
    into a 7-dim vector by SUM then divide by sentence_length (the
    whole sentence's token count, not this bucket's size -- see
    RATIONALE.md). is_negator/is_contrast are left at 0 unconditionally:
    bucket members never include a negator or contrast-marker token
    (see _assign_buckets), so those two columns are always dead here.

    When redefine_negated/redefine_modified is set, the
    negation_sensitivity / degree_modifier column accumulates a
    per-token product computed before pooling (RATIONALE.md's pooling-
    formula fix) instead of a raw column average.
    """
    vec = np.zeros(N_EMBED_COLS, dtype=np.float64)
    if sentence_length == 0:
        return vec

    for idx, prev_idx in members:
        vec[VALENCE] += E[idx, VALENCE]
        if redefine_negated:
            vec[NEGATION_SENSITIVITY] += -E[idx, NEGATION_SENSITIVITY] * E[idx, VALENCE]
        else:
            vec[NEGATION_SENSITIVITY] += E[idx, NEGATION_SENSITIVITY]
        if redefine_modified:
            prev_degree = E[prev_idx, DEGREE_MODIFIER] if prev_idx is not None else 0.0
            vec[DEGREE_MODIFIER] += E[idx, VALENCE] * prev_degree
        else:
            vec[DEGREE_MODIFIER] += E[idx, DEGREE_MODIFIER]
        vec[CERTAINTY] += E[idx, CERTAINTY]
        vec[CONTENT_NEGATOR] += E[idx, CONTENT_NEGATOR]

    return vec / sentence_length


def encode_sentence(sentence: str, word_to_index: dict, E: np.ndarray) -> np.ndarray:
    """Encodes one sentence into the fixed 49-dim vector composition
    layer 1 consumes. Pure function: no file I/O, no randomness, no
    dependence on anything but its three arguments. Returns an all-zero
    vector for a sentence that tokenizes to nothing.
    """
    tokens = tokenize(sentence)
    out = np.zeros(N_ENCODER_OUTPUTS, dtype=np.float64)
    sentence_length = len(tokens)
    if sentence_length == 0:
        return out

    indices = _token_indices(tokens, word_to_index)
    scope = _negation_scope(tokens, indices, E)
    split_at = _find_split(indices, E)

    if split_at is None:
        pre_idx, post_idx = indices, []
        pre_scope: list[bool] = scope
        post_scope: list[bool] = []
        has_contrast = 0.0
    else:
        pre_idx, post_idx = indices[:split_at], indices[split_at + 1 :]
        pre_scope, post_scope = scope[:split_at], scope[split_at + 1 :]
        has_contrast = 1.0

    pre_buckets = _assign_buckets(pre_idx, pre_scope, E)
    post_buckets = _assign_buckets(post_idx, post_scope, E) if post_idx else _EMPTY_BUCKETS

    for clause_prefix, clause_buckets in (("pre", pre_buckets), ("post", post_buckets)):
        for bucket_name in ("plain", "negated", "modified"):
            base = BLOCK_BASE[f"{clause_prefix}_{bucket_name}"]
            out[base : base + N_EMBED_COLS] = _pool_block(
                clause_buckets[bucket_name],
                E,
                sentence_length,
                redefine_negated=(bucket_name == "negated"),
                redefine_modified=(bucket_name == "modified"),
            )

    out[SCALAR_INDEX["has_contrast"]] = has_contrast
    out[SCALAR_INDEX["pre_len_frac"]] = len(pre_idx) / sentence_length
    out[SCALAR_INDEX["post_len_frac"]] = len(post_idx) / sentence_length
    out[SCALAR_INDEX["num_negators"]] = float(
        sum(1 for idx in indices if E[idx, IS_NEGATOR] == 1.0)
    )
    out[SCALAR_INDEX["has_exclaim"]] = 1.0 if "!" in tokens else 0.0
    out[SCALAR_INDEX["has_question"]] = 1.0 if "?" in tokens else 0.0
    out[SCALAR_INDEX["sentence_length"]] = float(sentence_length)

    return out
