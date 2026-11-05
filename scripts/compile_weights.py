"""Mechanically compile authoring/*.yaml + vocab/vocab.json into
weights/weights.npz.

This script contains no sentiment judgment of its own -- see
authoring/lexicon_families.py, authoring/structural_words.yaml, and
authoring/composition_weights.yaml for every judgment call it reads.
It only:
  1. Builds the (vocab_size, 7) embedding table E, placing each
     authored lexicon.yaml / structural_words.yaml value into the row
     for its word's vocab index. Column assignment fixed in
     handnet/columns.py.
  2. Assembles composition layer 1's weight matrix/bias vector, the
     fusion weight vector/alpha/beta, and output_scale/output_bias from
     composition_weights.yaml's sparse per-unit maps. Unit order fixed
     in handnet/units.py.
  3. Writes everything to weights/weights.npz as named arrays.

Same judgment-free-compiler relationship to its YAML inputs that
scripts/expand_lexicon.py has to authoring/lexicon_families.py.

Run: python scripts/compile_weights.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from handnet.columns import (  # noqa: E402
    CERTAINTY,
    CONTENT_NEGATOR,
    DEGREE_MODIFIER,
    IS_CONTRAST,
    IS_NEGATOR,
    NEGATION_SENSITIVITY,
    N_EMBED_COLS,
    VALENCE,
)
from handnet.units import N_ENCODER_OUTPUTS, N_UNITS, UNIT_NAMES  # noqa: E402

VOCAB_PATH = os.path.join(ROOT, "vocab", "vocab.json")
LEXICON_PATH = os.path.join(ROOT, "authoring", "lexicon.yaml")
STRUCTURAL_PATH = os.path.join(ROOT, "authoring", "structural_words.yaml")
COMPOSITION_PATH = os.path.join(ROOT, "authoring", "composition_weights.yaml")
OUT_PATH = os.path.join(ROOT, "weights", "weights.npz")


def _load(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh) if path.endswith(".json") else yaml.safe_load(fh)


def build_embedding_table(vocab: dict, lexicon: dict, structural: dict) -> np.ndarray:
    size = vocab["size"]
    index_of = {w: info["index"] for w, info in vocab["words"].items()}
    E = np.zeros((size, N_EMBED_COLS), dtype=np.float64)

    for word, entry in lexicon["words"].items():
        idx = index_of.get(word)
        if idx is None:
            continue  # lexicon.yaml is already vocab-filtered; defensive only
        E[idx, VALENCE] = entry["valence"]
        E[idx, NEGATION_SENSITIVITY] = entry["sensitivity"]

    for word in structural["negators"]:
        idx = index_of.get(word)
        if idx is not None:
            E[idx, IS_NEGATOR] = 1.0

    for word in structural["contrast_markers"]:
        idx = index_of.get(word)
        if idx is not None:
            E[idx, IS_CONTRAST] = 1.0

    for word, value in structural["degree_modifiers"].items():
        idx = index_of.get(word)
        if idx is not None:
            E[idx, DEGREE_MODIFIER] = value

    for word, value in structural["certainty"].items():
        idx = index_of.get(word)
        if idx is not None:
            E[idx, CERTAINTY] = value

    for word in structural["content_negators"]:
        idx = index_of.get(word)
        if idx is not None:
            E[idx, CONTENT_NEGATOR] = 1.0

    return E


def build_composition_layer_1(composition: dict) -> tuple[np.ndarray, np.ndarray]:
    units = composition["composition_layer_1"]
    W = np.zeros((N_UNITS, N_ENCODER_OUTPUTS), dtype=np.float64)
    b = np.zeros(N_UNITS, dtype=np.float64)
    for row, name in enumerate(UNIT_NAMES):
        spec = units[name]
        for idx, weight in spec["weights"].items():
            W[row, int(idx)] = weight
        b[row] = spec["bias"]
    return W, b


def build_fusion(composition: dict) -> tuple[np.ndarray, float, float]:
    fusion = composition["fusion"]
    f = np.array([fusion["weights"][name] for name in UNIT_NAMES], dtype=np.float64)
    return f, float(fusion["alpha"]), float(fusion["beta"])


def main() -> None:
    vocab = _load(VOCAB_PATH)
    lexicon = _load(LEXICON_PATH)
    structural = _load(STRUCTURAL_PATH)
    composition = _load(COMPOSITION_PATH)

    E = build_embedding_table(vocab, lexicon, structural)
    comp1_W, comp1_b = build_composition_layer_1(composition)
    fusion_w, alpha, beta = build_fusion(composition)
    output_scale = float(composition["output"]["output_scale"])
    output_bias = float(composition["output"]["output_bias"])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(
        OUT_PATH,
        E=E,
        comp1_W=comp1_W,
        comp1_b=comp1_b,
        fusion_w=fusion_w,
        alpha=alpha,
        beta=beta,
        output_scale=output_scale,
        output_bias=output_bias,
    )

    n_embed_nonzero = int(np.count_nonzero(E))
    n_comp1_nonzero = int(np.count_nonzero(comp1_W)) + int(np.count_nonzero(comp1_b))
    n_fusion_nonzero = int(np.count_nonzero(fusion_w))
    total_params = E.size + comp1_W.size + comp1_b.size + fusion_w.size + 2 + 2

    print(f"embedding table E: shape {E.shape}, {n_embed_nonzero} nonzero entries")
    print(
        f"composition layer 1: weights {comp1_W.shape} + bias {comp1_b.shape}, "
        f"{n_comp1_nonzero} nonzero (weights+bias)"
    )
    print(
        f"fusion weights: shape {fusion_w.shape}, {n_fusion_nonzero} nonzero; "
        f"alpha={alpha}, beta={beta}"
    )
    print(f"output: scale={output_scale}, bias={output_bias}")
    print(f"total parameter count: {int(total_params)}")
    print(f"written -> {OUT_PATH}")


if __name__ == "__main__":
    main()
