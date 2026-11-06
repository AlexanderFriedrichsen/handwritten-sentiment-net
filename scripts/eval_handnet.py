"""The one official, one-shot scoring run of the frozen hand-written net
against the disclosed SST-2 validation split (872 rows) -- PROTOCOL.md
section 8. Never run this more than once and keep both numbers; if you
genuinely must re-run it (e.g. environment repro check), compare against
the already-committed results/handnet_eval.json rather than overwriting
it, which is why this script refuses to overwrite an existing results
file unless you pass --force.

Depends on results/label_audit.json already existing (run
scripts/audit_labels.py first): this script reads that file's
`mismatch_found` verdict and applies the disclosed polarity correction
if and only if the audit found one. It does not re-decide the
convention question itself -- that decision belongs entirely to the
one allowed Phase 2 revision, logged in authoring/RATIONALE.md, and
this script would rather fail loudly than duplicate or drift from it.

`labels_seen` for the frozen net itself is 0 -- it never saw a label
during authoring (see FREEZE.md, freeze.json). Computing accuracy/F1
unavoidably *reads* the 872 validation labels for scoring, exactly as
computing any model's final metric reads its test labels; that is not
what `labels_seen` tracks (model/weight-influencing label exposure),
and this script's output says so explicitly to head off the obvious
confusion.

Run: python scripts/eval_handnet.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from handnet.net import SentimentNet  # noqa: E402
from scripts.eval_common import binary_metrics, fetch_raw  # noqa: E402
from scripts.verify_freeze import verify_freeze  # noqa: E402

RESULTS_PATH = ROOT / "results" / "handnet_eval.json"
AUDIT_PATH = ROOT / "results" / "label_audit.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing results/handnet_eval.json instead of refusing",
    )
    args = parser.parse_args()

    if RESULTS_PATH.exists() and not args.force:
        print(
            f"refusing to overwrite existing {RESULTS_PATH} -- this is meant to be "
            "scored exactly once. Pass --force if you deliberately want to replace it "
            "(e.g. a reproducibility re-check), and keep the original for comparison.",
            file=sys.stderr,
        )
        return 1

    if not AUDIT_PATH.exists():
        print(
            f"{AUDIT_PATH} does not exist -- run scripts/audit_labels.py first. "
            "The official score must not be computed before the one allowed "
            "label-convention audit has run and logged its finding.",
            file=sys.stderr,
        )
        return 1

    verify_freeze()
    audit = json.loads(AUDIT_PATH.read_text())
    flip = bool(audit["mismatch_found"])

    net = SentimentNet()
    val_df = fetch_raw("validation")
    sentences = val_df["sentence"].tolist()
    true_label = val_df["label"].to_numpy()

    proba = np.array([net.predict_proba(s) for s in sentences])
    pred = (proba > 0.5).astype(int)
    if flip:
        pred = 1 - pred

    metrics = binary_metrics(true_label, pred)

    result = {
        "split": "validation",
        "n_rows": int(len(sentences)),
        "labels_seen": {
            "value": 0,
            "note": (
                "The frozen net never saw a label during authoring (see FREEZE.md, "
                "freeze.json) -- this is the headline zero-label result. Computing "
                "accuracy/F1 below necessarily reads this split's 872 labels for "
                "scoring, exactly as every baseline's own final metric reads its "
                "scored split; that label-reading-for-scoring is not what "
                "`labels_seen` tracks across this project (model/weight-influencing "
                "label exposure -- training, tuning, selection). See "
                "scripts/run_baselines.py and RESULTS.md for the baselines' own "
                "labels_seen accounting, which is nonzero by construction."
            ),
        },
        "label_convention": {
            "mismatch_found_by_audit": flip,
            "correction_applied": "flipped" if flip else "none",
            "source": "results/label_audit.json, authoring/RATIONALE.md",
        },
        "metrics": metrics,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print(f"scored {len(sentences)} validation rows (correction: {result['label_convention']['correction_applied']})")
    print(f"accuracy: {metrics['accuracy']:.4f}  f1: {metrics['f1']:.4f}")
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
