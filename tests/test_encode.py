"""Mechanism/wiring tests for handnet.encode.

These check that the encoder computes the structural transformation
specified in authoring/RATIONALE.md -- never a sentiment judgment, and
never by looking at what the full net predicts (net.py is not even
imported here). Per RATIONALE.md's integrity rule 4, a failing test in
this file may only fix an implementation bug, never retune a weight
value in authoring/*.yaml; any such fix must be logged there.
"""

import json
import os

import numpy as np
import pytest

from handnet.columns import DEGREE_MODIFIER, NEGATION_SENSITIVITY, N_EMBED_COLS, VALENCE
from handnet.encode import encode_sentence
from handnet.units import BLOCK_BASE, DEAD_INDICES, N_ENCODER_OUTPUTS, SCALAR_INDEX

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def vocab():
    with open(os.path.join(ROOT, "vocab", "vocab.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    return {w: info["index"] for w, info in data["words"].items()}


@pytest.fixture(scope="module")
def E(_compiled_weights):
    with np.load(os.path.join(ROOT, "weights", "weights.npz")) as data:
        return data["E"]


def _vec(sentence, vocab, E):
    return encode_sentence(sentence, vocab, E)


def test_output_shape(vocab, E):
    out = _vec("this movie was good .", vocab, E)
    assert out.shape == (N_ENCODER_OUTPUTS,)


def test_empty_sentence_is_all_zero(vocab, E):
    out = _vec("   ", vocab, E)
    assert np.all(out == 0.0)


def test_dead_indices_always_zero(vocab, E):
    sentences = [
        "this movie was not very good but the acting was great",
        "an utterly brilliant , surprisingly moving film",
        "nobody could call this anything but a disaster",
        "the plot is n't bad , however the pacing drags",
    ]
    for s in sentences:
        out = _vec(s, vocab, E)
        for i in DEAD_INDICES:
            assert out[i] == 0.0, f"index {i} nonzero for: {s!r}"


def test_negator_moves_valence_from_plain_to_negated_bucket(vocab, E):
    plain = _vec("good", vocab, E)
    negated = _vec("not good", vocab, E)

    plain_base = BLOCK_BASE["pre_plain"]
    negated_base = BLOCK_BASE["pre_negated"]

    assert plain[plain_base + VALENCE] > 0.0
    assert negated[plain_base + VALENCE] == 0.0
    assert negated[negated_base + VALENCE] != 0.0


def test_negated_sensitivity_column_is_the_redefined_product(vocab, E):
    idx_good = vocab["good"]
    valence = E[idx_good, VALENCE]
    sensitivity = E[idx_good, NEGATION_SENSITIVITY]
    assert valence > 0.0 and sensitivity > 0.0  # test assumption on lexicon.yaml

    out = _vec("not good", vocab, E)
    base = BLOCK_BASE["pre_negated"]
    sentence_length = 2  # "not", "good"
    expected = (-sensitivity * valence) / sentence_length
    assert out[base + NEGATION_SENSITIVITY] == pytest.approx(expected)


def test_degree_modifier_opens_modified_bucket_and_redefines_column(vocab, E):
    idx_good = vocab["good"]
    idx_very = vocab["very"]
    valence = E[idx_good, VALENCE]
    degree = E[idx_very, DEGREE_MODIFIER]
    assert valence > 0.0 and degree != 0.0  # test assumption

    out = _vec("very good", vocab, E)
    base = BLOCK_BASE["pre_modified"]
    sentence_length = 2
    expected = (valence * degree) / sentence_length
    assert out[base + DEGREE_MODIFIER] == pytest.approx(expected)

    plain_base = BLOCK_BASE["pre_plain"]
    assert out[plain_base + VALENCE] == 0.0  # both tokens claimed by "modified"


def test_negated_takes_precedence_over_modified(vocab, E):
    # "not very good": "very" and "good" both fall inside the negation
    # window opened by "not", so neither should land in pre_modified.
    out = _vec("not very good", vocab, E)
    modified_base = BLOCK_BASE["pre_modified"]
    assert np.all(out[modified_base : modified_base + N_EMBED_COLS] == 0.0)
    negated_base = BLOCK_BASE["pre_negated"]
    assert out[negated_base + VALENCE] != 0.0


def test_contrast_marker_splits_clause_and_excludes_itself(vocab, E):
    out = _vec("good but bad", vocab, E)
    assert out[SCALAR_INDEX["has_contrast"]] == 1.0

    pre_base = BLOCK_BASE["pre_plain"]
    post_base = BLOCK_BASE["post_plain"]
    assert out[pre_base + VALENCE] != 0.0
    assert out[post_base + VALENCE] != 0.0

    # "but" itself (1 of 3 tokens) must land in neither clause.
    total_len_frac = out[SCALAR_INDEX["pre_len_frac"]] + out[SCALAR_INDEX["post_len_frac"]]
    assert total_len_frac == pytest.approx(2 / 3)


def test_negation_scope_resets_at_contrast_marker(vocab, E):
    # The window opened by "not" must not cross "but" into the post
    # clause: "bad" should land in post_plain, not post_negated.
    out = _vec("not good but bad", vocab, E)
    assert out[BLOCK_BASE["post_negated"] + VALENCE] == 0.0
    assert out[BLOCK_BASE["post_plain"] + VALENCE] != 0.0


def test_strong_punctuation_resets_negation_scope(vocab, E):
    # "." closes the window opened by "not" before "good" is reached.
    out = _vec("not bad . good", vocab, E)
    plain_base = BLOCK_BASE["pre_plain"]
    negated_base = BLOCK_BASE["pre_negated"]
    # "good" must have landed in plain, not negated.
    assert out[plain_base + VALENCE] > 0.0
    # only "bad" (negative) is in the negated bucket, not "good" too.
    assert out[negated_base + VALENCE] < 0.0


def test_scalars(vocab, E):
    out = _vec("what a great movie !", vocab, E)
    assert out[SCALAR_INDEX["has_exclaim"]] == 1.0
    assert out[SCALAR_INDEX["has_question"]] == 0.0
    assert out[SCALAR_INDEX["sentence_length"]] == 5.0


def test_unknown_word_dilutes_but_adds_no_signal(vocab, E):
    out_known = _vec("good", vocab, E)
    out_unknown = _vec("good zzzznotaword", vocab, E)
    plain_base = BLOCK_BASE["pre_plain"]
    assert out_unknown[plain_base + VALENCE] == pytest.approx(
        out_known[plain_base + VALENCE] * 1 / 2
    )
