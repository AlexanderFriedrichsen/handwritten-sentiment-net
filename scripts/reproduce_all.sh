#!/usr/bin/env bash
# Regenerates every Phase 2 evaluation artifact end-to-end from a clean
# clone: venv + pinned requirements.txt, freeze verification, the
# label-convention audit, the one-shot official validation score, the
# sklearn baseline sweep, the results plot, and the full test suite.
#
# Never touches handnet/, weights/weights.npz, authoring/*.yaml, or
# freeze.json -- those are the frozen artifacts this script verifies
# against, never regenerates. Only results/*.json and figures/*.png are
# ever removed or rewritten here.
#
# **Why results/handnet_eval.json and results/baselines.json are force-
# regenerated on every run, but results/label_audit.json is NOT:** both
# scripts/eval_handnet.py and scripts/run_baselines.py refuse to
# overwrite their own results file without --force -- a guard against an
# accidental, unnoticed re-run silently replacing a reported number, not
# against this script's own explicit, visible "regenerate everything"
# job. Since results/*.json is committed (data/ is the only gitignored
# directory -- see .gitignore), every clone already has these files
# present, so this script removes and regenerates them deliberately on
# every run; `git diff --stat results/ figures/` afterward shows whether
# anything actually changed. scripts/audit_labels.py has the same
# --force guard, but ALSO unconditionally appends its finding to
# authoring/RATIONALE.md's append-only log every time it runs
# successfully -- force-regenerating it on every routine reproduction
# run would spam that log with duplicate entries for a finding that
# never changes (the audit reads only the frozen net and a fixed, seeded
# TRAIN slice -- nothing here is expected to drift between runs). So
# this script treats an existing results/label_audit.json as "already
# audited, skip" rather than force-regenerating it; if you deliberately
# need to re-run the audit itself, remove results/label_audit.json and
# its corresponding RATIONALE.md entry by hand first.
#
# SMOKE=1 ./scripts/reproduce_all.sh runs the baseline sweep with tiny
# budgets/seeds for a fast correctness check (seconds, not ~70s+) and
# skips the plot step, which needs real budgets to be worth drawing.
#
# Run: ./scripts/reproduce_all.sh
#      SMOKE=1 ./scripts/reproduce_all.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=.venv/bin/python

step() {
  echo
  echo "=== $1 ==="
}

step "1/7 Python environment (.venv + pinned requirements.txt)"
if [ ! -x "$PY" ]; then
  python3 -m venv .venv
fi
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements.txt

step "2/7 verify freeze"
"$PY" scripts/verify_freeze.py

step "3/7 label-convention audit (scripts/audit_labels.py)"
if [ -f results/label_audit.json ]; then
  echo "results/label_audit.json already exists -- skipping re-run."
  echo "Its finding is logged once into authoring/RATIONALE.md's append-only log;"
  echo "re-running would append a duplicate entry. To genuinely regenerate it,"
  echo "remove results/label_audit.json and its RATIONALE.md entry by hand first."
else
  "$PY" scripts/audit_labels.py
fi

step "4/7 official one-shot validation score (scripts/eval_handnet.py)"
rm -f results/handnet_eval.json
"$PY" scripts/eval_handnet.py

step "5/7 sklearn baseline sweep (scripts/run_baselines.py)"
rm -f results/baselines.json
if [ "${SMOKE:-0}" = "1" ]; then
  "$PY" scripts/run_baselines.py --smoke
else
  "$PY" scripts/run_baselines.py
fi

if [ "${SMOKE:-0}" = "1" ]; then
  step "6/7 plot (skipped under SMOKE=1 -- smoke budgets have nothing meaningful to plot)"
else
  step "6/7 plot (scripts/plot_results.py)"
  "$PY" scripts/plot_results.py
fi

step "7/7 full test suite"
"$PY" -m pytest -q

step "done"
echo "Regenerated: results/handnet_eval.json, results/baselines.json"
if [ "${SMOKE:-0}" != "1" ]; then
  echo "  and figures/results.png"
fi
echo "results/label_audit.json was left untouched if it already existed (see step 3/7)."
echo "Compare against the committed versions with: git diff --stat results/ figures/"
