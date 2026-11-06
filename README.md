# handwritten-sentiment-net

Round 2 of [handwritten-poker-net](https://github.com/AlexanderFriedrichsen/handwritten-poker-net): a hand-authored numpy neural net for SST-2 binary sentiment, built to properly replicate/validate a viral claim that a hand-written net with sparse params hit high accuracy with zero labels -- a claim whose "one-shot or hill climbing?" question was never answered. This project answers that by construction: one process authors and freezes every weight without ever reading a label, a separate process evaluates afterward.

**Status: weights authored and frozen (tag `frozen-v1`); evaluated (Phase 2
complete).** Phase 1 authored every paragraph below and every frozen weight
without ever seeing a label, and never will -- that guarantee is unaffected
by what follows. This status line and the first bullet below were added
afterward by the separate Phase 2 evaluation process `PROTOCOL.md`
describes, which likewise never touched a frozen weight. See:

- [`RESULTS.md`](RESULTS.md) -- the frozen net's one-shot validation score,
  labeled sklearn baselines trained on its identical encoding, and the one
  disclosed, reasoned deviation from `PROTOCOL.md`'s literal audit
  instructions.
- [`PROTOCOL.md`](PROTOCOL.md) -- the visibility rules, the two-phase split, and the one narrowly-accounted-for exception to "never touch a frozen weight again."
- [`FREEZE.md`](FREEZE.md) / [`freeze.json`](freeze.json) -- SHA-256 hashes of every file that determines the net's behavior, plus the integrity attestation.
- [`authoring/RATIONALE.md`](authoring/RATIONALE.md) -- the timestamped, append-only log of every design and weight decision, written as it happened.

**Result summary:** the frozen net scores 0.5103 accuracy / 0.3190 F1 on
the one-shot, zero-label SST-2 validation pass (872 rows) -- just 0.0011
above this split's majority-class baseline (0.5092), driven by a strong,
uneven bias toward predicting "negative" (79.0% of predictions) rather
than by the net behaving like a majority-class guesser outright. sklearn
baselines trained on the net's own frozen encoding pass this accuracy
quickly -- `logistic_regression` by the smallest budget tested (16
labels), `hist_gbt` by 64, `mlp` by 256 -- and all three converge to a
0.728-0.734 band by the full 64049-label training pool, well short of
the cited BERT-base reference (92.7% dev accuracy, outside pretraining).
Full numbers, the baseline sweep, the one disclosed protocol deviation,
and caveats are in [`RESULTS.md`](RESULTS.md).

## What this is

A genuine numpy neural network (fixed-shape linear algebra, no
per-example branching downstream of a label-free sentence encoder):
tokenize -> vocabulary lookup -> clause/negation-aware pooling into a
49-number vector -> a 12-unit hidden layer -> contrast-gated fusion ->
sigmoid. Roughly 100k total parameters, thousands of them non-zero,
every one typed by hand from lexical/structural judgment -- no
gradient descent, no fitting, no search, no external sentiment
lexicon, no pretrained model or embedding anywhere in its construction.
