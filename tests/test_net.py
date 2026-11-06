"""Mechanism/wiring tests for handnet.net.

These check that the forward pass implements the fusion formula
documented in authoring/RATIONALE.md's "Composition/fusion/output
design" entry against hand-computed expected values -- never whether a
prediction "looks right" for any sentence's sentiment. Per the task's
integrity rules, no test anywhere in this suite inspects or reports an
accuracy figure, or changes a weight because output looked wrong; a
different worker evaluates accuracy later, on a split this process
never loads.
"""

import numpy as np
import pytest

from handnet.net import SentimentNet
from handnet.units import N_ENCODER_OUTPUTS, UNIT_NAMES

# Independent of handnet.net's own _PRE_IDX/_POST_IDX/_SIGNAL_IDX, so a
# mistake in that internal grouping would not also be baked into the
# test's expectation.
_PRE = ("plain_sentiment_pre", "negated_sentiment_pre", "modified_sentiment_pre")
_POST = ("plain_sentiment_post", "negated_sentiment_post", "modified_sentiment_post")
_SIGNAL = tuple(u for u in UNIT_NAMES if u not in _PRE + _POST)
_PRE_IDX = [UNIT_NAMES.index(u) for u in _PRE]
_POST_IDX = [UNIT_NAMES.index(u) for u in _POST]
_SIGNAL_IDX = [UNIT_NAMES.index(u) for u in _SIGNAL]


@pytest.fixture(scope="module")
def net(_compiled_weights):
    return SentimentNet()


def test_composition_1_matches_hand_computed_tanh(net):
    x = np.zeros(N_ENCODER_OUTPUTS)
    x[0] = 2.0  # index 0 == plain_sentiment_pre's only weighted input (weight 1.0, bias 0.0)
    h = net.composition_1(x)
    unit = UNIT_NAMES.index("plain_sentiment_pre")
    assert h[unit] == pytest.approx(np.tanh(2.0))


def test_fuse_matches_documented_formula_no_contrast(net):
    h = np.arange(1, 13, dtype=np.float64)  # h1..h12 = 1..12, arbitrary but known
    raw = net.fuse(h, has_contrast=0.0)

    f = net.fusion_w
    pre_term = sum(f[i] * h[i] for i in _PRE_IDX)
    post_term = sum(f[i] * h[i] for i in _POST_IDX)
    signal_term = sum(f[i] * h[i] for i in _SIGNAL_IDX)
    expected = 1.0 * pre_term + 0.0 * post_term + signal_term
    assert raw == pytest.approx(expected)


def test_fuse_matches_documented_formula_with_contrast(net):
    h = np.arange(1, 13, dtype=np.float64)
    raw = net.fuse(h, has_contrast=1.0)

    f = net.fusion_w
    pre_term = sum(f[i] * h[i] for i in _PRE_IDX)
    post_term = sum(f[i] * h[i] for i in _POST_IDX)
    signal_term = sum(f[i] * h[i] for i in _SIGNAL_IDX)
    expected = net.alpha * pre_term + net.beta * post_term + signal_term
    assert raw == pytest.approx(expected)
    assert net.alpha == pytest.approx(0.4)
    assert net.beta == pytest.approx(1.6)


def test_output_layer_is_pure_passthrough(net):
    # output_scale=1.0, output_bias=0.0 are frozen identity defaults
    # (RATIONALE.md): forward() must equal fuse()'s raw_score exactly,
    # not some rescaled version of it.
    assert net.output_scale == 1.0
    assert net.output_bias == 0.0

    encoded = np.zeros(N_ENCODER_OUTPUTS)
    h = net.composition_1(encoded)
    raw_score = net.fuse(h, has_contrast=0.0)
    assert net.forward("") == pytest.approx(raw_score)


def test_predict_proba_is_a_valid_probability_for_any_input(net):
    for sentence in ["good", "bad", "not good", "but", "", "   ", "a" * 50]:
        p = net.predict_proba(sentence)
        assert 0.0 <= p <= 1.0


def test_predict_proba_is_sigmoid_of_forward(net):
    sentence = "not very good but the ending was great"
    logit = net.forward(sentence)
    p = net.predict_proba(sentence)
    assert p == pytest.approx(1.0 / (1.0 + np.exp(-logit)))


def test_forward_is_deterministic(net):
    sentence = "an utterly forgettable mess , despite a strong lead performance"
    assert net.forward(sentence) == net.forward(sentence)
