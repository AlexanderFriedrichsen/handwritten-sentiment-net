"""Shared Phase 2 (evaluation) helpers: nothing here is imported by, or
affects, anything under handnet/, authoring/, weights/, or vocab/ -- it
only reads the already-frozen net and the labeled SST-2 data Phase 1
never saw. Every top-level script still calls
scripts.verify_freeze.verify_freeze() itself, first, before importing
anything from this module that touches a label.

Owns three things so every Phase 2 script agrees on them without
duplicating logic:
  1. Fetching + caching the labeled `train` and `validation` SST-2
     splits (never `test` -- its labels are hidden GLUE placeholders,
     and PROTOCOL.md never licenses reading it anyway).
  2. Encoding a split through the frozen, identical encode_sentence()
     every baseline is required to train on (PROTOCOL.md §7).
  3. The one deterministic, seeded partition of the TRAIN split's row
     indices into three disjoint groups -- audit / tune / pool -- so
     scripts/audit_labels.py's disclosed audit slice, and
     scripts/run_baselines.py's one-time hyperparameter-search slice,
     never overlap each other or the budget-sampling pool.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

HF_BASE = "https://huggingface.co/datasets/stanfordnlp/sst2/resolve/main/data"
SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
}

# Deterministic, disclosed partition of the TRAIN split's row indices.
# See audit_labels.py (uses AUDIT) and run_baselines.py (uses TUNE, then
# samples every budget -- including "full train" -- from POOL).
AUDIT_SLICE_SIZE = 300
TUNE_SLICE_SIZE = 3000
SLICE_SEED = 20261105


def fetch_raw(split: str) -> pd.DataFrame:
    """Downloads (once, cached under the gitignored data/raw/) and
    returns the `sentence` + `label` columns of the given SST-2 split.
    Phase 2 is unrestricted on labels (PROTOCOL.md §6-7), unlike
    scripts/fetch_sst2.py's Phase-1, label-free, train-only fetch.
    """
    if split not in SPLIT_FILES:
        raise ValueError(
            f"unknown split {split!r}; this project only ever reads "
            "'train' and 'validation' -- 'test' labels are hidden GLUE "
            "placeholders and PROTOCOL.md never licenses reading them."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{split}.parquet"
    if not cache_path.exists():
        url = f"{HF_BASE}/{SPLIT_FILES[split]}"
        print(f"Downloading {split} split from: {url}")
        tmp_path = RAW_DIR / f"_download_{split}.parquet"
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, cache_path)

    df = pd.read_parquet(cache_path, columns=["sentence", "label"])
    assert list(df.columns) == ["sentence", "label"], list(df.columns)
    bad_labels = sorted(set(df["label"].unique()) - {0, 1})
    assert not bad_labels, f"unexpected label values in {split}: {bad_labels}"
    return df.reset_index(drop=True)


def get_encoded(split: str, net) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (X, y, sentences) for `split`, X being the (n, 49) matrix
    of the frozen net's own encode_sentence() output -- the identical
    encoding PROTOCOL.md §7 requires every baseline to train on.
    Encoding is cached (gitignored, data/encoded_<split>.npz) since it
    is the same deterministic computation every script needs.
    """
    df = fetch_raw(split)
    sentences = df["sentence"].tolist()
    labels = df["label"].to_numpy(dtype=np.int64)

    cache_path = DATA_DIR / f"encoded_{split}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached["X"].shape[0] == len(sentences) and np.array_equal(cached["y"], labels):
            return cached["X"], cached["y"], sentences

    X = np.stack([net.encode(s) for s in sentences]).astype(np.float64)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, y=labels)
    return X, labels, sentences


def reserve_slices(n_total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically partitions range(n_total) (TRAIN split row
    indices) into three disjoint index arrays: (audit_idx, tune_idx,
    pool_idx), of sizes (AUDIT_SLICE_SIZE, TUNE_SLICE_SIZE,
    n_total - AUDIT_SLICE_SIZE - TUNE_SLICE_SIZE). Seeded by the fixed,
    disclosed SLICE_SEED so every script that calls this agrees on the
    same three groups without coordinating directly or persisting
    anything to disk.
    """
    rng = np.random.default_rng(SLICE_SEED)
    perm = rng.permutation(n_total)
    audit_idx = perm[:AUDIT_SLICE_SIZE]
    tune_idx = perm[AUDIT_SLICE_SIZE : AUDIT_SLICE_SIZE + TUNE_SLICE_SIZE]
    pool_idx = perm[AUDIT_SLICE_SIZE + TUNE_SLICE_SIZE :]
    return audit_idx, tune_idx, pool_idx


def binary_metrics(y_true, y_pred) -> dict:
    """Accuracy + F1 (PROTOCOL.md §8's headline + companion metric),
    plus precision/recall/confusion matrix for full auditability.
    confusion_matrix layout: [[TN, FP], [FN, TP]] (rows = true 0/1,
    cols = predicted 0/1), SST-2's convention of label 1 = positive.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }
