"""Numpy forward pass: encoder output -> composition layer 1 (tanh) ->
fusion / composition layer 2 (contrast-gated linear combination) ->
output (identity scale/bias, sigmoid).

A genuine matrix-multiply network, not per-example branching: the only
sentence-dependent control flow in the whole pipeline lives in
handnet.encode (clause splitting, negation scope, bucket assignment),
which reduces every sentence to a fixed-size 49-number vector; from
there on every sentence runs the identical linear-algebra code path.
See authoring/RATIONALE.md's "Composition/fusion/output design" entry
for the fusion formula implemented in `fuse` below.
"""

from __future__ import annotations

import json
import os

import numpy as np

from handnet.encode import encode_sentence
from handnet.units import SCALAR_INDEX, UNIT_NAMES

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WEIGHTS_PATH = os.path.join(ROOT, "weights", "weights.npz")
DEFAULT_VOCAB_PATH = os.path.join(ROOT, "vocab", "vocab.json")

_PRE_UNITS = ("plain_sentiment_pre", "negated_sentiment_pre", "modified_sentiment_pre")
_POST_UNITS = ("plain_sentiment_post", "negated_sentiment_post", "modified_sentiment_post")
_SIGNAL_UNITS = tuple(u for u in UNIT_NAMES if u not in _PRE_UNITS + _POST_UNITS)

_PRE_IDX = [UNIT_NAMES.index(u) for u in _PRE_UNITS]
_POST_IDX = [UNIT_NAMES.index(u) for u in _POST_UNITS]
_SIGNAL_IDX = [UNIT_NAMES.index(u) for u in _SIGNAL_UNITS]


class SentimentNet:
    """Loads a compiled weights.npz and vocab.json once; call
    .predict_proba(sentence) (or the intermediate .encode / .forward
    steps) as many times as needed.
    """

    def __init__(
        self, weights_path: str = DEFAULT_WEIGHTS_PATH, vocab_path: str = DEFAULT_VOCAB_PATH
    ):
        with np.load(weights_path) as data:
            self.E = data["E"]
            self.comp1_W = data["comp1_W"]
            self.comp1_b = data["comp1_b"]
            self.fusion_w = data["fusion_w"]
            self.alpha = float(data["alpha"])
            self.beta = float(data["beta"])
            self.output_scale = float(data["output_scale"])
            self.output_bias = float(data["output_bias"])

        with open(vocab_path, encoding="utf-8") as fh:
            vocab = json.load(fh)
        self.word_to_index = {w: info["index"] for w, info in vocab["words"].items()}

    def encode(self, sentence: str) -> np.ndarray:
        return encode_sentence(sentence, self.word_to_index, self.E)

    def composition_1(self, encoded: np.ndarray) -> np.ndarray:
        return np.tanh(self.comp1_W @ encoded + self.comp1_b)

    def fuse(self, h: np.ndarray, has_contrast: float) -> float:
        """raw_score = pre_weight*(f1 h1+f2 h2+f3 h3)
        + post_weight*(f4 h4+f5 h5+f6 h6) + f7 h7+...+f12 h12, with
        pre_weight = 1 - has_contrast*(1-alpha), post_weight =
        has_contrast*beta. See RATIONALE.md for why this particular
        gate: pre_weight==1/post_weight==0 when there is no contrast
        clause, pre_weight==alpha/post_weight==beta when there is.
        """
        pre_weight = 1.0 - has_contrast * (1.0 - self.alpha)
        post_weight = has_contrast * self.beta

        pre_term = sum(self.fusion_w[i] * h[i] for i in _PRE_IDX)
        post_term = sum(self.fusion_w[i] * h[i] for i in _POST_IDX)
        signal_term = sum(self.fusion_w[i] * h[i] for i in _SIGNAL_IDX)

        return pre_weight * pre_term + post_weight * post_term + signal_term

    def forward(self, sentence: str) -> float:
        """Returns the raw logit (pre-sigmoid)."""
        encoded = self.encode(sentence)
        h = self.composition_1(encoded)
        has_contrast = float(encoded[SCALAR_INDEX["has_contrast"]])
        raw_score = self.fuse(h, has_contrast)
        return raw_score * self.output_scale + self.output_bias

    def predict_proba(self, sentence: str) -> float:
        logit = self.forward(sentence)
        return float(1.0 / (1.0 + np.exp(-logit)))
