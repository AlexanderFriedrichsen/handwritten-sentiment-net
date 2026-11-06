"""Mechanism tests for scripts/compile_weights.py's output: confirm the
compiled weights/weights.npz matches the structural facts fixed in
authoring/RATIONALE.md (shapes, total parameter count, the dead-index
invariant, the all-biases-zero / identity-output invariant). Never
inspects sentiment correctness -- see test_encode.py for that.
"""

import os

import numpy as np

from handnet.units import DEAD_INDICES, N_ENCODER_OUTPUTS, N_UNITS

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_PATH = os.path.join(ROOT, "weights", "weights.npz")


def _load():
    with np.load(WEIGHTS_PATH) as data:
        return {k: data[k] for k in data.files}


def test_shapes():
    w = _load()
    assert w["E"].ndim == 2
    assert w["E"].shape[1] == 7
    assert w["comp1_W"].shape == (N_UNITS, N_ENCODER_OUTPUTS)
    assert w["comp1_b"].shape == (N_UNITS,)
    assert w["fusion_w"].shape == (N_UNITS,)


def test_total_param_count_matches_rationale():
    w = _load()
    total = w["E"].size + w["comp1_W"].size + w["comp1_b"].size + w["fusion_w"].size + 2 + 2
    assert int(total) == 100786


def test_composition_layer_1_nonzero_count_matches_rationale():
    w = _load()
    assert int(np.count_nonzero(w["comp1_W"])) == 31


def test_fusion_nonzero_count_matches_rationale():
    w = _load()
    assert int(np.count_nonzero(w["fusion_w"])) == 12


def test_no_unit_weights_a_dead_index():
    w = _load()
    for i in DEAD_INDICES:
        assert np.all(w["comp1_W"][:, i] == 0.0), f"a composition unit weights dead index {i}"


def test_all_biases_are_zero():
    w = _load()
    assert np.all(w["comp1_b"] == 0.0)
    assert float(w["output_bias"]) == 0.0


def test_output_layer_is_identity_default():
    w = _load()
    assert float(w["output_scale"]) == 1.0
    assert float(w["output_bias"]) == 0.0


def test_embedding_table_never_touches_unk_row():
    w = _load()
    assert np.all(w["E"][0] == 0.0)
