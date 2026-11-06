# Freeze

**Frozen at**: 2026-11-05T19:26:00Z, branch `fm/sentiment-author`, tag
`frozen-v1`.

**`labels_seen = 0`.** No SST-2 label of any kind -- validation, test,
or train -- was read, loaded, printed, or computed over by this process
before, during, or after authoring the weights below. Data preparation
read only the raw `sentence` text of the **train** split. See
`PROTOCOL.md` for the full visibility rules this freeze certifies
against, and `authoring/RATIONALE.md` for the timestamped, append-only
log of every weight decision.

## What is frozen

Every file that determines the net's behavior -- the hand-authored
weight sources, the compiled weight artifact, the vocabulary, and every
line of encoder/forward-pass/compile code -- is hashed below (SHA-256).
After this commit, none of these files is modified again for the
lifetime of this repo, with exactly one narrow, accounted-for exception
(`PROTOCOL.md` §6: a label-convention audit only, never a retune).

The machine-readable version of this table, plus the exact recipe for
the manifest hash, is in `freeze.json`.

| File | SHA-256 |
|---|---|
| `authoring/composition_weights.yaml` | `70496a8a5f19b18e6677c17ec6362963e7ff5bd3b91abdde7d5e5f6c7b54dc5c` |
| `authoring/lexicon.yaml` | `fbf04bb8bd3509950f577f9f541f0fa76c93c1532c472f44a6dddc11d8cb82a8` |
| `authoring/lexicon_families.py` | `a579884e32af699624fdc3b50aae47f51545eda9a933d153963d60eb025c5c75` |
| `authoring/structural_words.yaml` | `e386bca2c64a23dcc737870cb1daa49fb415adf16de92773d22d46f052ad77a2` |
| `handnet/__init__.py` | `6001b3fa49f2cd267168a61468b87d9bdcd3df83969d614b81007214ea4d95c8` |
| `handnet/columns.py` | `5cbb538f028afaf75e98614d7ba54c1d946cfe9971f3ac5c4d0c705376e99e20` |
| `handnet/encode.py` | `88e67f32df2d0e0c09d8008c656d7dd981d1fa5418a2e0c2fd5d121cd6e75d7f` |
| `handnet/net.py` | `db01cd8de857d963d2079bf646f44750e6a80f303b02e664838da3e268a0473a` |
| `handnet/tokenize.py` | `a8e81326d50fd625d845ad9189590f5768468ed913479fa5afea1cd0cb80da97` |
| `handnet/units.py` | `f80a328460c8ada863292fdbdf3d14533a33a5bfd234390a4958b78567ff7b76` |
| `scripts/compile_weights.py` | `4f88c2fe796737911ea4cea1f21a5423ab24454326b7d3bd35423cdb7ca52bf0` |
| `scripts/expand_lexicon.py` | `64eebd521365d6e7bf7b452e41411ab62db5f745d727b58a0c00b7f36ef1b679` |
| `vocab/vocab.json` | `fdacf7787d0236e44d21da4e47ebd47974e70ca8e6a63fb412704bf3fe7a761a` |
| `weights/weights.npz` | `b6b7e4bca79db7746316c46bd97e1726f6b5c6353c5b7b93be0244af91514400` |

**Manifest hash** (sha256 over the sorted `path hash` lines above --
exact recipe in `freeze.json`):

```
e6a45b66ac8696514d8d7b62db67d22b61d0a7713a3174c5c0027c62eb0c1f34
```

To re-verify any single file: `sha256sum <path>` and compare. To
re-verify the whole set at once:

```
for f in authoring/composition_weights.yaml authoring/lexicon.yaml \
         authoring/lexicon_families.py authoring/structural_words.yaml \
         handnet/__init__.py handnet/columns.py handnet/encode.py \
         handnet/net.py handnet/tokenize.py handnet/units.py \
         scripts/compile_weights.py scripts/expand_lexicon.py \
         vocab/vocab.json weights/weights.npz; do
  sha256sum "$f"
done | awk '{print $2, $1}' | sort | sha256sum
```

Deliberately excluded from the manifest, with reasons (also in
`freeze.json`'s `excluded_from_manifest`): `scripts/build_vocab.py` and
`scripts/fetch_sst2.py` are one-time construction procedures whose
entire effect is already captured by the hashed `vocab/vocab.json` and
the (gitignored, not part of the running model) raw fetched text;
`scripts/report_params.py` is read-only reporting, never consulted by
the compiler, encoder, or net.

## Integrity attestation

Covering the hard integrity rules this project ran under for the
entirety of Phase 1 (full text and reasoning in `authoring/RATIONALE.md`
and `PROTOCOL.md`):

1. **No label access.** Never true otherwise -- confirmed by construction:
   the only SST-2 data this process ever loaded was the train split's
   `sentence` column.
2. **No label-conditioned statistics, fitting, optimizer, or search.**
   Every weight in `authoring/*.yaml` was typed by hand.
3. **No external sentiment lexicons.** Every valence/sensitivity/degree/
   certainty value is this process's own lexical judgment, not copied
   from AFINN, VADER, Hu-Liu, SentiWordNet, or any other existing
   resource.
4. **No output-driven tuning.** The net was run only inside mechanism
   tests, checked against hand-computed expected values from the
   documented formulas -- never against a sentence's sentiment "looking
   right." All 27 tests passed on first run; zero weights were retuned
   after any test.
5. **Contemporaneous rationale log.** `authoring/RATIONALE.md` was
   appended to, timestamped, as each decision was made, and has never
   been rewritten after the fact.

## What happens next

Phase 2 (evaluation) is performed by a separate process, against the
full labeled SST-2 test set, per `PROTOCOL.md`. This process does not
perform that evaluation and will not see its result.
