"""Renders figures/results.png from results/baselines.json and
results/handnet_eval.json -- the one chart PROTOCOL.md's evaluation
brief asks for: baseline accuracy vs. label budget, with the frozen
hand-net and a cited (not trained) pretrained reference model as two
horizontal reference lines.

**Why horizontal reference lines, not points at some x position:**
Both the hand-net and the literature reference have no "label budget"
that belongs on this plot's x-axis -- the hand-net never saw a label at
all (labels_seen=0, which has no placement on a log-x axis: log(0) is
undefined), and the literature number is cited from outside pretraining
on data this project never touches. Representing either as a point
at a made-up x position would imply a budget comparison that does not
exist. A full-width horizontal line makes the actual comparison this
plot supports explicit instead: "how many labels does training
directly on this net's own 49-dim encode_sentence() output need before
it matches/beats each reference."

**Literature reference, cited not trained** (PROTOCOL.md's "optional
citation" allowance): BERT-BASE, 92.7% accuracy on the SST-2 **Dev**
set -- Devlin, Chang, Lee, Toutanova (2018), "BERT: Pre-training of
Deep Bidirectional Transformers for Language Understanding",
arXiv:1810.04805, Table 5 ("Effect of Pre-training Tasks", Section
5.1), the row explicitly labeled BERT_BASE. Deliberately NOT the
paper's own headline Table 1 number (93.5) -- that table's caption
states those are "GLUE Test results, scored by the evaluation server"
against GLUE's hidden test set, not the dev set this project's own
872-row validation split is (see scripts/eval_common.py fetch_raw,
HF dataset stanfordnlp/sst2, split "validation" -- the canonical GLUE
SST-2 dev set). Citing the dev-set figure keeps the comparison to the
same split on both sides; citing the test-server figure would not.
Table 6 of the same paper reports a close but distinct 92.9 for a
model matching BERT_BASE's architecture, averaged over 5 fine-tuning
restarts, under a different protocol -- disclosed here rather than
silently picking whichever number looks better.

Run: python scripts/plot_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASELINES_PATH = ROOT / "results" / "baselines.json"
HANDNET_PATH = ROOT / "results" / "handnet_eval.json"
OUT_PATH = ROOT / "figures" / "results.png"

CHANCE_ACCURACY = 0.5
LIT_REFERENCE_ACCURACY = 0.927
LIT_REFERENCE_LABEL = (
    "BERT-base, SST-2 dev (Devlin et al. 2018, Table 5) -- outside pretraining, cited not trained"
)

FAMILY_STYLE = {
    "logistic_regression": {"color": "#1f77b4", "marker": "o", "label": "logistic regression"},
    "hist_gbt": {"color": "#ff7f0e", "marker": "s", "label": "gradient-boosted trees"},
    "mlp": {"color": "#2ca02c", "marker": "^", "label": "small MLP"},
}


def main() -> int:
    if not BASELINES_PATH.exists() or not HANDNET_PATH.exists():
        print(
            f"both {BASELINES_PATH} and {HANDNET_PATH} must exist first -- run "
            "scripts/eval_handnet.py and scripts/run_baselines.py before this script.",
            file=sys.stderr,
        )
        return 1

    baselines = json.loads(BASELINES_PATH.read_text())
    handnet = json.loads(HANDNET_PATH.read_text())

    if baselines["smoke"]:
        print(
            f"{BASELINES_PATH} was written with --smoke (tiny budgets/seeds) -- its numbers "
            "are a correctness check, not a result worth plotting. Re-run "
            "scripts/run_baselines.py without --smoke first.",
            file=sys.stderr,
        )
        return 1

    rows = baselines["rows"]
    budgets = sorted({r["budget"] for r in rows})

    fig, ax = plt.subplots(figsize=(9, 6))

    for family, style in FAMILY_STYLE.items():
        means, stds = [], []
        for budget in budgets:
            accs = [r["accuracy"] for r in rows if r["family"] == family and r["budget"] == budget]
            means.append(float(np.mean(accs)))
            stds.append(float(np.std(accs)))
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(budgets, means, color=style["color"], marker=style["marker"], label=style["label"], zorder=3)
        ax.fill_between(budgets, means - stds, means + stds, color=style["color"], alpha=0.15, zorder=2)

    handnet_acc = handnet["metrics"]["accuracy"]
    ax.axhline(
        handnet_acc,
        color="black",
        linestyle="--",
        linewidth=1.5,
        zorder=4,
        label=f"frozen hand-net, labels_seen=0 (accuracy={handnet_acc:.4f})",
    )
    ax.axhline(
        LIT_REFERENCE_ACCURACY,
        color="purple",
        linestyle="-.",
        linewidth=1.5,
        zorder=4,
        label=f"{LIT_REFERENCE_LABEL} (accuracy={LIT_REFERENCE_ACCURACY:.3f})",
    )
    ax.axhline(CHANCE_ACCURACY, color="gray", linestyle=":", linewidth=1, zorder=1, label="chance (0.5)")

    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("labels_seen (training labels, log scale)")
    ax.set_ylabel("accuracy on the 872-row SST-2 validation split")
    ax.set_title("Frozen zero-label hand-net vs. label-budgeted sklearn baselines")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
