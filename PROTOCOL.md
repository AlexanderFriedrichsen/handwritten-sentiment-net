# Protocol

This repo follows the round-2 protocol drafted in the benchmark/protocol
pick report (`report.md` §4; not included in this repo, see the round-1
project for the sibling document). This file is the repo-local,
authoritative restatement of that protocol: the concrete rules this
specific build commits to, so a reader can audit this repo without
needing the external report. Where this file and that report ever
disagree, this file governs this repo.

The single thing this protocol exists to make auditable: the original
viral claim it replicates/validates (a hand-written net, zero labels,
97.8% accuracy on an undisclosed task) was publicly challenged --
"was it one-shot or was it hill climbing?" -- and never defended. This
project answers that question by construction, not by assertion.

## Two phases, two different people

- **Phase 1 -- author and freeze** (this repo, this branch, complete as
  of the `frozen-v1` tag). One process designs the architecture, builds
  the vocabulary, and hand-authors every weight, then freezes and
  hashes the result. This process never reads a label.
- **Phase 2 -- evaluate** (a separate process, not started by, and with
  no access to the authoring process's reasoning beyond what is
  committed here). Runs the frozen net against the full labeled SST-2
  test set and reports accuracy/F1 alongside the baselines in §6-7
  below. The author of Phase 1 must never see this number.

The whole point of the split is that Phase 1 cannot have hill-climbed
against an outcome it never saw, and Phase 2 cannot retroactively
launder that guarantee by letting the same process see labels and then
touch weights.

## 1. Visibility rules (Phase 1, already applied)

Phase 1 was permitted to read: the SST-2 dataset's own documentation
(task definition, label semantics, official split sizes), the raw
unlabeled `sentence` text of the **train** split only, and general
linguistic/lexicographic knowledge not derived from this dataset's
labels.

Phase 1 never read, loaded, printed, or computed anything from: any
SST-2 label, any per-example annotation, the validation split, the test
split, or any dataset viewer/statistics page. No pretrained model,
pretrained embedding, or external sentiment lexicon (AFINN, VADER,
Hu-Liu, SentiWordNet, or similar) was read or used as a source of any
weight -- every valence/sensitivity/degree/certainty value in
`authoring/*.yaml` is this process's own lexical judgment, logged at
the time it was written in `authoring/RATIONALE.md`.

## 2. Ban on label-conditioned statistics (permanent, not just Phase 1)

Prohibited, for the lifetime of this project, and confirmed never done:
any class-conditional statistic over labels; any mutual-information- or
chi-squared-with-label feature ranking; any fitting, optimizer, or
search of any kind against any labeled data, including synthetic or
self-written examples used to "check if a weight feels right." Nothing
in this repo was constructed this way -- see `authoring/RATIONALE.md`'s
contemporaneous entries for every weight-group decision.

## 3. Label-free vocabulary / feature fixing

The vocabulary (`vocab/vocab.json`, built by `scripts/build_vocab.py`)
was fixed from the **train** split's raw unlabeled sentences only,
before any weight existed. The exact procedure (frequency cutoff,
resulting size) is documented in `authoring/RATIONALE.md`'s
`2026-11-05T18:15Z` entry. The 7-column embedding schema
(`handnet/columns.py`) and the encoder's 49-number output layout
(`handnet/units.py`) were likewise fixed as architecture before any
weight value existed (`RATIONALE.md`'s `18:18Z` entries).

## 4. Judgment-based weight authoring + auditability

Every weight was assigned by this process's own semantic judgment,
with a contemporaneous, timestamped, append-only rationale log
(`authoring/RATIONALE.md`) -- never rewritten after the fact. No
gradient descent, no backprop, no search, no iterative refinement
against any feedback signal (labeled, synthetic, or this net's own
output) was used anywhere. Mechanism/wiring tests
(`tests/test_compile_weights.py`, `tests/test_encode.py`,
`tests/test_net.py`) were written and run against hand-computed
expected values derived from the documented formulas -- never to judge
whether a prediction looked right for a sentence's sentiment. All 27
tests passed on first run; this process made zero post-test weight
retunings, so there is nothing further to log under this rule.

## 5. Freeze + hash

Weight authoring is complete. The full set of files that determine the
net's behavior -- authored weight sources, compiled weights, vocabulary,
and all encoder/forward-pass/compile code -- is hashed (SHA-256, per
file, plus one manifest hash over all of them) in `freeze.json`, with
the human-readable summary in `FREEZE.md`. The freeze commit is tagged
`frozen-v1`. **No SST-2 label was read by this process before, during,
or after this freeze.** `labels_seen = 0` at freeze -- see `FREEZE.md`
for the full attestation.

After the freeze commit, `weights/`, `vocab/vocab.json`, every file
under `handnet/`, `scripts/compile_weights.py`, `scripts/expand_lexicon.py`,
and every file under `authoring/` except `RATIONALE.md` itself are
immutable for the lifetime of this repo, with exactly one narrow,
accounted-for exception: §6 below.

## 6. Separate evaluator, one disclosed revision

Phase 2's evaluation script is not written or modified by the Phase 1
authoring process, and is run against the full labeled SST-2 test set.

Exactly one Phase 2 revision is permitted, and it is narrower than it
sounds: a **label-definition/convention audit only**, against the
disclosed SST-2 **validation** split (872 rows) -- checking for a
systematic mismatch between this process's assumptions and the
dataset's actual label convention (e.g., confirming label `1` means
positive in this specific release, not retuning any weight because
accuracy looked low). This is not a license to iterate on weights based
on observed accuracy.

If this revision is used: the finding (mismatch or no mismatch -- a
clean "no mismatch found" must be reported plainly, not omitted) and
its justification are logged in `authoring/RATIONALE.md` at the time of
the audit, any resulting fix is logged the same way, and
`labels_seen` for that revision is recorded as `0 (authoring budget) +
872 (validation-split size)` = **872**, applied only to that revision's
reported numbers -- never silently folded into the frozen, zero-label
`labels_seen = 0` headline result.

If this revision is not used, `labels_seen` remains `0` for the single
reported hand-written-net result.

## 7. Baselines (Phase 2)

All baselines (logistic regression / linear model, gradient-boosted
trees, a small from-scratch MLP) train on the identical frozen encoding
this net uses -- same vocabulary, same features. Label budgets are
swept across multiple points spanning small to large (e.g. 16, 64, 256,
1000, 5000, full-train), each run across multiple random seeds, with
mean +/- variance reported. `labels_seen = budget + dev_size` for any
baseline that used a disclosed dev slice for hyperparameter selection.
Baseline hyperparameters are chosen via a disclosed, bounded search on
one fixed labeled sample, then locked in identically across all budgets
and seeds. Exactly one additional, clearly-separated strong reference
model (pretrained-and-fine-tuned or few-shot-prompted) may be shown,
explicitly labeled as using outside pretraining this net and the
from-scratch baselines do not have.

## 8. Metrics

Accuracy as the headline metric, plus F1 alongside it. Seeds and
variance are reported for every trained point. `labels_seen` is
reported alongside every single number this project publishes,
including the hand-written net's.

## 9. Status

Phase 1: **complete**, frozen at tag `frozen-v1`
(`labels_seen = 0`). Phase 2: **not started** by this process -- a
separate worker evaluates from here.
