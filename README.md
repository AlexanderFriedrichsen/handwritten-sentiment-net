# handwritten-sentiment-net

Round 2 of [handwritten-poker-net](https://github.com/AlexanderFriedrichsen/handwritten-poker-net): a hand-authored numpy neural net for SST-2 binary sentiment, built to properly replicate/validate a viral claim that a hand-written net with sparse params hit high accuracy with zero labels -- a claim whose "one-shot or hill climbing?" question was never answered. This project answers that by construction: one process authors and freezes every weight without ever reading a label, a separate process evaluates afterward.

**Status: weights authored and frozen (tag `frozen-v1`), evaluation pending.** This repo makes no results claim of any kind -- no accuracy, no comparison to baselines -- because the authoring process that wrote this README has not seen and will not see that number. See:

- [`PROTOCOL.md`](PROTOCOL.md) -- the visibility rules, the two-phase split, and the one narrowly-accounted-for exception to "never touch a frozen weight again."
- [`FREEZE.md`](FREEZE.md) / [`freeze.json`](freeze.json) -- SHA-256 hashes of every file that determines the net's behavior, plus the integrity attestation.
- [`authoring/RATIONALE.md`](authoring/RATIONALE.md) -- the timestamped, append-only log of every design and weight decision, written as it happened.

## What this is

A genuine numpy neural network (fixed-shape linear algebra, no
per-example branching downstream of a label-free sentence encoder):
tokenize -> vocabulary lookup -> clause/negation-aware pooling into a
49-number vector -> a 12-unit hidden layer -> contrast-gated fusion ->
sigmoid. Roughly 100k total parameters, thousands of them non-zero,
every one typed by hand from lexical/structural judgment -- no
gradient descent, no fitting, no search, no external sentiment
lexicon, no pretrained model or embedding anywhere in its construction.
