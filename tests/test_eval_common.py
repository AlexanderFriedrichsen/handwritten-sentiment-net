"""Tests for scripts/eval_common.py's pure, no-I/O helpers: reserve_slices
and binary_metrics. Deliberately excludes fetch_raw/get_encoded -- those
need network access and the compiled net, and are exercised end-to-end
by actually running scripts/audit_labels.py, scripts/eval_handnet.py,
and scripts/run_baselines.py instead of being mocked here.
"""

import numpy as np

from scripts.eval_common import AUDIT_SLICE_SIZE, TUNE_SLICE_SIZE, binary_metrics, reserve_slices

N_TOTAL = 67349  # the real SST-2 train split's row count, for a realistic-scale check


def test_reserve_slices_sizes():
    audit_idx, tune_idx, pool_idx = reserve_slices(N_TOTAL)
    assert len(audit_idx) == AUDIT_SLICE_SIZE
    assert len(tune_idx) == TUNE_SLICE_SIZE
    assert len(pool_idx) == N_TOTAL - AUDIT_SLICE_SIZE - TUNE_SLICE_SIZE
    assert len(audit_idx) + len(tune_idx) + len(pool_idx) == N_TOTAL


def test_reserve_slices_pairwise_disjoint():
    audit_idx, tune_idx, pool_idx = reserve_slices(N_TOTAL)
    a, t, p = set(audit_idx.tolist()), set(tune_idx.tolist()), set(pool_idx.tolist())
    assert not (a & t)
    assert not (a & p)
    assert not (t & p)
    assert a | t | p == set(range(N_TOTAL))


def test_reserve_slices_deterministic_across_calls():
    r1 = reserve_slices(N_TOTAL)
    r2 = reserve_slices(N_TOTAL)
    for arr1, arr2 in zip(r1, r2):
        assert np.array_equal(arr1, arr2)


def test_binary_metrics_perfect_predictions():
    y = np.array([0, 1, 1, 0, 1])
    m = binary_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 3]]


def test_binary_metrics_all_wrong_predictions():
    y_true = np.array([0, 1, 1, 0])
    y_pred = 1 - y_true
    m = binary_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert m["f1"] == 0.0


def test_binary_metrics_confusion_matrix_layout():
    # rows = true 0/1, cols = predicted 0/1: [[TN, FP], [FN, TP]]
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 1])
    m = binary_metrics(y_true, y_pred)
    tn, fp, fn, tp = 2, 1, 1, 3
    assert m["confusion_matrix"] == [[tn, fp], [fn, tp]]
    assert m["n"] == 7
