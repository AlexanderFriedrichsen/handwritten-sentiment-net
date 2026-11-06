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
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASELINES_PATH = ROOT / "results" / "baselines.json"
HANDNET_PATH = ROOT / "results" / "handnet_eval.json"
OUT_PATH = ROOT / "figures" / "results.png"
X_THREAD_OUT_PATH = ROOT / "figures" / "x_labels_vs_handnet.png"

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

# Short forms for the X-thread chart's direct line-end labels -- that chart
# has no legend box, so every label is read at the font size used for a
# phone screen, where FAMILY_STYLE's full names ("gradient-boosted trees")
# cost more width than the tight full-budget cluster (see
# plot_x_thread_chart) leaves room for.
X_THREAD_FAMILY_SHORT_LABEL = {
    "logistic_regression": "logistic regression",
    "hist_gbt": "boosted trees",
    "mlp": "small MLP",
}


def _spread_for_labels(true_values: list[float], min_gap: float) -> list[float]:
    """Returns label y-positions for `true_values` that preserve each
    value's rank but are evenly spread at least `min_gap` apart, centered
    on their mean. Used only for the X-thread chart's full-budget labels,
    which cluster within 0.0064 of each other -- closer together than one
    text line at phone-readable font size, so some of the three must be
    nudged off their true y or their text would overlap. Spreading by
    rank (not hand-picked constants) keeps this correct if a future
    re-run of scripts/run_baselines.py shifts the numbers or their order.
    """
    n = len(true_values)
    order = sorted(range(n), key=lambda i: true_values[i])
    center = sum(true_values) / n
    spread_sorted = [center + (rank - (n - 1) / 2) * min_gap for rank in range(n)]
    result = [0.0] * n
    for rank, idx in enumerate(order):
        result[idx] = spread_sorted[rank]
    return result


def plot_x_thread_chart(baselines: dict, handnet: dict) -> None:
    """Renders figures/x_labels_vs_handnet.png: a single-panel, 1600x900
    (16:9) restyling of the same results/baselines.json + results/
    handnet_eval.json data main() plots into figures/results.png, built
    for docs/x-thread.md post 1/ instead of PROTOCOL.md's evaluation
    brief -- larger fonts, lines labeled directly at their ends instead
    of a legend box, and the frozen hand-net drawn as a bold reference
    line with its own headline number in the label.

    Omits the BERT-base literature line plotted in figures/results.png:
    docs/x-thread.md's posts cite that number in prose (if at all) rather
    than on this chart, and at 0.927 it would force most of this chart's
    vertical space to sit empty between it and everything being compared
    here (0.50-0.73), fighting the "phone-readable fonts" requirement.
    """
    rows = baselines["rows"]
    budgets = sorted({r["budget"] for r in rows})
    handnet_acc = handnet["metrics"]["accuracy"]

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

    means_by_family = {}
    for family, style in FAMILY_STYLE.items():
        means, stds = [], []
        for budget in budgets:
            accs = [r["accuracy"] for r in rows if r["family"] == family and r["budget"] == budget]
            means.append(float(np.mean(accs)))
            stds.append(float(np.std(accs)))
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        means_by_family[family] = means_arr
        ax.plot(budgets, means_arr, color=style["color"], linewidth=3.5, zorder=3, solid_capstyle="round")
        ax.fill_between(budgets, means_arr - stds_arr, means_arr + stds_arr, color=style["color"], alpha=0.18, zorder=2)

    ax.axhline(handnet_acc, color="black", linewidth=3.5, zorder=4)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(10.0, budgets[-1])
    ax.set_xticks(budgets)
    ax.set_xticklabels(["16", "64", "256", "1,000", "5,000", "64,049\n(full)"], fontsize=17)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(0.44, 0.87)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_xlabel("labels seen (log scale)", fontsize=19)
    ax.set_ylabel("accuracy on the 872-row validation split", fontsize=19)
    fig.suptitle(
        "A few labels beat hand-written weights on movie reviews",
        fontsize=27,
        fontweight="bold",
        x=0.5,
        y=0.97,
    )
    ax.grid(True, which="major", axis="y", alpha=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Reserve a fixed right-hand margin (in figure space, not data space) for
    # the direct line-end labels below -- their text lives past x=1.0 in
    # axes-fraction coordinates, outside the plotted budgets entirely, so it
    # cannot overlap the lines/bands and does not move if the data does.
    # fig.tight_layout() does not know about those out-of-axes text artists
    # and would otherwise let them run past the canvas edge, so margins are
    # set explicitly here instead.
    fig.subplots_adjust(left=0.065, right=0.68, top=0.90, bottom=0.14)
    yaxis_transform = ax.get_yaxis_transform()  # x: axes fraction [0, 1]; y: data

    full_budget_means = [float(means_by_family[family][-1]) for family in FAMILY_STYLE]
    label_ys = _spread_for_labels(full_budget_means, min_gap=0.035)
    for (family, style), label_y in zip(FAMILY_STYLE.items(), label_ys):
        true_y = float(means_by_family[family][-1])
        ax.annotate(
            f"{X_THREAD_FAMILY_SHORT_LABEL[family]}: {true_y:.1%}",
            xy=(budgets[-1], true_y),
            xycoords="data",
            xytext=(1.03, label_y),
            textcoords=yaxis_transform,
            fontsize=16,
            fontweight="bold",
            color=style["color"],
            va="center",
            ha="left",
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=style["color"], linewidth=1.3, shrinkA=0, shrinkB=4),
        )

    ax.text(
        1.03,
        handnet_acc,
        "Hand-written by Claude,\n0 labels: 51.0%",
        transform=yaxis_transform,
        color="black",
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )
    ax.text(
        1.03,
        0.5,
        "coin flip: 50%",
        transform=yaxis_transform,
        color="gray",
        fontsize=15,
        va="top",
        ha="left",
        clip_on=False,
    )

    cross_x, cross_y = budgets[0], float(means_by_family["logistic_regression"][0])
    ax.annotate(
        f"logistic regression already beats\nthe hand-net by {cross_x} labels",
        xy=(cross_x, cross_y),
        xycoords="data",
        xytext=(budgets[1] * 1.6, 0.635),
        fontsize=16,
        fontweight="bold",
        color=FAMILY_STYLE["logistic_regression"]["color"],
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
        arrowprops=dict(
            arrowstyle="-",
            color=FAMILY_STYLE["logistic_regression"]["color"],
            linewidth=1.5,
            shrinkA=0,
            shrinkB=6,
        ),
    )

    X_THREAD_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(X_THREAD_OUT_PATH, dpi=100)
    print(f"wrote {X_THREAD_OUT_PATH}")


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
    # Anchored below the BERT-base reference line (0.927, only ~0.07 above the
    # y=1.0 axes top -- too little headroom for the legend box) and above the
    # baseline curves' rise, which stays under 0.68 through this x range.
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.78), fontsize=8)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")

    plot_x_thread_chart(baselines, handnet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
