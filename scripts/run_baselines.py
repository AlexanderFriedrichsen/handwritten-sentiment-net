"""Trained sklearn baselines on the identical frozen 49-dim
encode_sentence() output (PROTOCOL.md section 7), swept across label
budgets and seeds, for direct accuracy/F1 comparison against the
frozen net's one-shot official score (results/handnet_eval.json) --
every row here is scored against that same 872-row validation split.

**Families**: logistic_regression (linear), hist_gbt =
HistGradientBoostingClassifier (trees), mlp = MLPClassifier (small
from-scratch-trained neural net). Picked to cover linear/tree/MLP
baseline families from scikit-learn alone, with no new heavy
dependency (no torch/xgboost/lightgbm) -- defensible at this data
scale (49 features, <= 64049 training rows, binary task) where
scikit-learn's own implementations are not a capability bottleneck.

**Hyperparameter search, bounded and one-time** (tune_hparams()): each
family's capacity/regularization hyperparameters (logistic_regression:
C; hist_gbt: learning_rate x max_leaf_nodes; mlp: hidden_layer_sizes x
alpha) are grid-searched ONCE against an 80/20 internal split of the
disjoint 3000-row tune slice (scripts/eval_common.reserve_slices),
with early_stopping disabled and a fixed, generous max_iter during the
search itself -- so the comparison across candidates reflects capacity
choices only, not incidentally-different stopping points. The winning
config per family is then locked in and reused, unchanged, for every
budget/seed fit below. This tuning cost (3000 labels) is disclosed
once, separately, in the output's hyperparameter_search block -- never
folded into any row's own labels_seen.

**Early stopping and labels_seen**: at real fit time, hist_gbt and mlp
use early_stopping=True with scikit-learn's own internal
validation_fraction, carved FROM the budget passed in -- not extra
rows read beyond it. So dev_size is 0 and labels_seen == budget for
every row, exactly: nothing beyond the stated budget is ever read for
either family. logistic_regression has no such mechanism at all (it is
solved to convergence via lbfgs); same accounting applies trivially.
At the smallest budgets (16, 64) hist_gbt/mlp's internal validation
carve is a handful of rows and the early-stopping signal from it is
correspondingly noisy -- a disclosed, honest property of the low-label
regime, not a bug.

**No class weighting**: SST-2 train is only mildly imbalanced (37569
positive / 29780 negative, ~56/44) -- too mild to warrant
inverse-frequency class weighting (contrast the severe ~2513:1 ratio
that motivated it in the handwritten-poker-net precedent). No class
weighting variant is run here; disclosed as a judgment call, not an
oversight.

Run --smoke first (seconds) to sanity-check correctness before
trusting the full sweep (several minutes, dominated by the top 1-2
budgets' mlp/hist_gbt fits).

Run: python scripts/run_baselines.py [--smoke] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from handnet.net import SentimentNet  # noqa: E402
from scripts.eval_common import binary_metrics, get_encoded, reserve_slices  # noqa: E402
from scripts.verify_freeze import verify_freeze  # noqa: E402

RESULTS_PATH = ROOT / "results" / "baselines.json"

TUNE_VAL_FRACTION = 0.2
TUNE_SPLIT_SEED = 777
TUNE_FIT_SEED = 0
TUNE_FIXED_MAX_ITER = 300

BUDGETS = [16, 64, 256, 1000, 5000, "full"]
SEEDS = [0, 1, 2, 3, 4]
REAL_FIT_MAX_ITER = 1000

LR_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]  # C (inverse regularization strength)
HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": mln}
    for lr in (0.3, 0.1, 0.03)
    for mln in (15, 31, 63)
]
MLP_GRID = [
    {"hidden_layer_sizes": hls, "alpha": alpha}
    for hls in ((32,), (64,), (32, 16))
    for alpha in (1e-4, 1e-3, 1e-2)
]


def sample_budget_idx(n_pool: int, budget: int, seed: int) -> np.ndarray:
    """Seeded, disclosed, reproducible draw of `budget` positions out of
    [0, n_pool) -- to be used as positions INTO pool_idx, never as raw
    TRAIN row indices directly. Mirrors the handwritten-poker-net
    precedent's sample_budget: a budget-and-seed-dependent RNG, not one
    shared seed reused verbatim across draws.
    """
    budget = min(budget, n_pool)
    rng = np.random.default_rng(900_001 + seed * 7919 + budget)
    return rng.choice(n_pool, size=budget, replace=False)


def fit_lr(X, y, C: float):
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
    model.fit(X, y)
    return model, int(model.n_iter_[0])


def fit_hgb(X, y, learning_rate: float, max_leaf_nodes: int, early_stopping: bool, max_iter: int, seed: int):
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        early_stopping=early_stopping,
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(X, y)
    return model, int(model.n_iter_)


def fit_mlp(X, y, hidden_layer_sizes: tuple, alpha: float, early_stopping: bool, max_iter: int, seed: int):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        early_stopping=early_stopping,
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(X, y)
    return model, int(model.n_iter_)


def tune_hparams(X_tune: np.ndarray, y_tune: np.ndarray) -> dict:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tune, y_tune, test_size=TUNE_VAL_FRACTION, random_state=TUNE_SPLIT_SEED, stratify=y_tune
    )

    best_lr_C, best_lr_acc = None, -1.0
    for C in LR_GRID:
        model, _ = fit_lr(X_tr, y_tr, C)
        acc = model.score(X_va, y_va)
        if acc > best_lr_acc:
            best_lr_C, best_lr_acc = C, acc
    print(f"  tuned logistic_regression: C={best_lr_C} val_acc={best_lr_acc:.4f}")

    best_hgb_cfg, best_hgb_acc = None, -1.0
    for cfg in HGB_GRID:
        model, _ = fit_hgb(
            X_tr, y_tr, cfg["learning_rate"], cfg["max_leaf_nodes"], False, TUNE_FIXED_MAX_ITER, TUNE_FIT_SEED
        )
        acc = model.score(X_va, y_va)
        if acc > best_hgb_acc:
            best_hgb_cfg, best_hgb_acc = cfg, acc
    print(f"  tuned hist_gbt: {best_hgb_cfg} val_acc={best_hgb_acc:.4f}")

    best_mlp_cfg, best_mlp_acc = None, -1.0
    for cfg in MLP_GRID:
        model, _ = fit_mlp(
            X_tr, y_tr, cfg["hidden_layer_sizes"], cfg["alpha"], False, TUNE_FIXED_MAX_ITER, TUNE_FIT_SEED
        )
        acc = model.score(X_va, y_va)
        if acc > best_mlp_acc:
            best_mlp_cfg, best_mlp_acc = cfg, acc
    print(f"  tuned mlp: {best_mlp_cfg} val_acc={best_mlp_acc:.4f}")

    return {
        "logistic_regression": {"C": best_lr_C, "tune_val_accuracy": best_lr_acc},
        "hist_gbt": {**best_hgb_cfg, "tune_val_accuracy": best_hgb_acc},
        "mlp": {"hidden_layer_sizes": list(best_mlp_cfg["hidden_layer_sizes"]), "alpha": best_mlp_cfg["alpha"],
                "tune_val_accuracy": best_mlp_acc},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny budgets/seeds for a fast correctness check")
    parser.add_argument("--force", action="store_true", help="overwrite an existing results/baselines.json")
    args = parser.parse_args()

    if RESULTS_PATH.exists() and not args.force:
        print(f"refusing to overwrite existing {RESULTS_PATH} -- pass --force to replace it.", file=sys.stderr)
        return 1

    verify_freeze()
    net = SentimentNet()

    X_train, y_train, _ = get_encoded("train", net)
    X_val, y_val, _ = get_encoded("validation", net)
    audit_idx, tune_idx, pool_idx = reserve_slices(len(y_train))
    n_pool = len(pool_idx)
    print(f"pool: {n_pool} rows (train={len(y_train)} minus audit={len(audit_idx)} minus tune={len(tune_idx)})")

    X_tune, y_tune = X_train[tune_idx], y_train[tune_idx]
    n_tune_tr = int(round(len(tune_idx) * (1 - TUNE_VAL_FRACTION)))
    n_tune_va = len(tune_idx) - n_tune_tr
    print(f"tuning hyperparameters on the {len(tune_idx)}-row tune slice "
          f"({n_tune_tr}/{n_tune_va} internal train/val split)...")
    hparams = tune_hparams(X_tune, y_tune)

    budget_specs = [16, 32] if args.smoke else BUDGETS
    seeds = [0, 1] if args.smoke else SEEDS
    resolved_budgets = [n_pool if b == "full" else b for b in budget_specs]

    rows = []
    t_start = time.time()
    for budget in resolved_budgets:
        for seed in seeds:
            local_idx = sample_budget_idx(n_pool, budget, seed)
            idx = pool_idx[local_idx]
            Xb, yb = X_train[idx], y_train[idx]
            classes_present = sorted(set(yb.tolist()))
            assert classes_present == [0, 1], (
                f"budget={budget} seed={seed} sampled only class(es) {classes_present} -- degenerate "
                "single-class budget, cannot fit a binary classifier. This is a real, disclosed edge "
                "case if it ever triggers -- investigate rather than silently skipping."
            )

            t0 = time.time()
            model, n_iter = fit_lr(Xb, yb, hparams["logistic_regression"]["C"])
            pred = model.predict(X_val)
            dt = time.time() - t0
            rows.append({
                "family": "logistic_regression",
                "budget": int(budget), "dev_size": 0, "labels_seen": int(budget), "seed": int(seed),
                "n_iterations": n_iter, "n_params": int(model.coef_.size + model.intercept_.size),
                "fit_seconds": dt,
                **binary_metrics(y_val, pred),
            })
            print(f"  budget={budget} seed={seed} logistic_regression: acc={rows[-1]['accuracy']:.4f} ({dt:.2f}s)")

            t0 = time.time()
            hgb_cfg = hparams["hist_gbt"]
            model, n_iter = fit_hgb(
                Xb, yb, hgb_cfg["learning_rate"], hgb_cfg["max_leaf_nodes"], True, REAL_FIT_MAX_ITER, seed
            )
            pred = model.predict(X_val)
            dt = time.time() - t0
            rows.append({
                "family": "hist_gbt",
                "budget": int(budget), "dev_size": 0, "labels_seen": int(budget), "seed": int(seed),
                "n_iterations": n_iter, "n_params": None,
                "fit_seconds": dt,
                **binary_metrics(y_val, pred),
            })
            print(f"  budget={budget} seed={seed} hist_gbt: acc={rows[-1]['accuracy']:.4f} iters={n_iter} ({dt:.2f}s)")

            t0 = time.time()
            mlp_cfg = hparams["mlp"]
            model, n_iter = fit_mlp(
                Xb, yb, tuple(mlp_cfg["hidden_layer_sizes"]), mlp_cfg["alpha"], True, REAL_FIT_MAX_ITER, seed
            )
            pred = model.predict(X_val)
            dt = time.time() - t0
            n_params = int(sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_))
            rows.append({
                "family": "mlp",
                "budget": int(budget), "dev_size": 0, "labels_seen": int(budget), "seed": int(seed),
                "n_iterations": n_iter, "n_params": n_params,
                "fit_seconds": dt,
                **binary_metrics(y_val, pred),
            })
            print(f"  budget={budget} seed={seed} mlp: acc={rows[-1]['accuracy']:.4f} iters={n_iter} ({dt:.2f}s)")

    output = {
        "schema_notes": {
            "budget": "number of TRAIN-pool rows the fit actually trains on (sampled from pool_idx, "
            "disjoint from the audit and tune slices -- see scripts/eval_common.py reserve_slices)",
            "dev_size": "always 0 -- hist_gbt/mlp use early_stopping=True with scikit-learn's own "
            "internal validation_fraction, carved FROM the budget itself (not extra rows read beyond "
            "it); logistic_regression has no such mechanism at all",
            "labels_seen": "budget + dev_size, i.e. always exactly equal to budget for every row here",
            "n_iterations": "lbfgs iterations (logistic_regression) / boosting rounds actually used at "
            "predict time (hist_gbt) / optimizer epochs actually run (mlp) -- reflects early stopping "
            "for hist_gbt/mlp, which varies legitimately by budget and seed",
            "n_params": "exact weight count for logistic_regression/mlp; null for hist_gbt, where an "
            "exact 'parameter count' for a tree ensemble is not a well-defined, comparable quantity -- "
            "see RESULTS.md for the capacity discussion instead",
            "evaluation_set": "every row is scored against the identical 872-row SST-2 validation split "
            "scripts/eval_handnet.py uses for the frozen net's official score, so accuracy/F1 numbers "
            "here are directly comparable to results/handnet_eval.json",
        },
        "hyperparameter_search": {
            "method": "one bounded grid search per family on the disjoint 3000-row tune slice (internal "
            f"{1 - TUNE_VAL_FRACTION:.0%}/{TUNE_VAL_FRACTION:.0%} train/val split, seed {TUNE_SPLIT_SEED}); "
            "early_stopping disabled during the search itself (fixed max_iter="
            f"{TUNE_FIXED_MAX_ITER}) so candidates are compared on capacity/regularization alone, not "
            "incidentally-different stopping points. Winners are locked in and reused unchanged for "
            "every budget/seed fit below, which instead uses early_stopping=True "
            f"(max_iter={REAL_FIT_MAX_ITER} ceiling) so each fit's own iteration count adapts to its "
            "own budget size.",
            "grids_searched": {
                "logistic_regression_C": LR_GRID,
                "hist_gbt_learning_rate_x_max_leaf_nodes": HGB_GRID,
                "mlp_hidden_layer_sizes_x_alpha": [
                    {"hidden_layer_sizes": list(c["hidden_layer_sizes"]), "alpha": c["alpha"]} for c in MLP_GRID
                ],
            },
            "winners": hparams,
            "labels_seen_for_tuning": int(len(tune_idx)),
            "tuning_cost_note": "This one-time tuning cost (3000 labels) is disclosed here, separately, "
            "and is NOT folded into any row's own labels_seen above -- a locked-in hyperparameter "
            "search is a one-time project cost, not a per-row cost.",
        },
        "class_weighting_note": "SST-2 train is mildly imbalanced (37569 positive / 29780 negative, "
        "~56/44) -- too mild to warrant inverse-frequency class weighting; no class-weighted variant "
        "is run (disclosed judgment call, not an oversight).",
        "n_pool": int(n_pool),
        "budget_specs": [str(b) for b in budget_specs],
        "budgets": [int(b) for b in resolved_budgets],
        "seeds": [int(s) for s in seeds],
        "smoke": bool(args.smoke),
        "rows": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {RESULTS_PATH}")
    print(f"total wall time: {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
