# Rationale log

Contemporaneous log of design and weight-authoring decisions, in the order
they were made. Appended-to as authoring proceeds; never rewritten after
the fact. Timestamps are UTC. This file is the auditable answer to "was it
one-shot or hill climbing?" (see report.md section 1) -- every weight-group
decision below was made before any weight value was fit, searched for, or
checked against output, and before any label was read.

---

## 2026-11-05T18:10Z -- Benchmark and integrity posture (inherited)

Benchmark is SST-2, per the standing protocol report
(`/home/alexp/.firstmate/data/handnet-fuzzy-pick/report.md`, sections 3-4).
This repo's brief is stricter than that report on one point and I am
following the brief: no external sentiment lexicon (AFINN/VADER/Hu-Liu/
SentiWordNet/etc.) is used as a source of any weight or word score, even
though report section 4.1 would have allowed it. Every valence, intensity,
and structural-word judgment below is my own semantic judgment, formed by
reasoning about the word/construction directly, not copied or derived from
any lexicon file, dataset statistic, or label.

## 2026-11-05T18:12Z -- Data fetch design

`scripts/fetch_sst2.py` downloads only the official `stanfordnlp/sst2`
TRAIN parquet (verified by HEAD request to resolve to a real file before
use), reads it with `pandas.read_parquet(path, columns=["sentence"])` so
pyarrow never decodes the `label` column's data pages, then deletes the
downloaded parquet immediately after that single read. Only the derived
`data/train_sentences.txt` (sentence text, one per line) persists, and
`data/` is gitignored. Validation and test splits are never requested. The
HF dataset *viewer* web page was never opened; only the documented resolve
URL pattern for the train parquet file was fetched directly. Result: 67,349
sentences, matching the train split size documented in the protocol report
exactly -- a sanity check that the right file was fetched, not a label
check (row count only, no label value was read to produce this count).

## 2026-11-05T18:14Z -- Tokenization

Inspected the raw sentence text (first 20 lines) and confirmed SST-2 is
already PTB-tokenized: lowercase, punctuation split into its own tokens
("," "." "n't" "'s" backtick/quote, "--"), hyphenated compounds kept whole.
Decision: tokenizer is a plain `.strip().lower().split()` -- no regex
needed, no judgment call, purely mechanical (`handnet/tokenize.py`). Shared
identically between vocabulary construction and the encoder so the two can
never drift apart.

## 2026-11-05T18:15Z -- Vocabulary construction

Computed token frequency over the unlabeled train sentences only (no label
anywhere in this computation). 14,816 unique types across 633,724 tokens
in 67,349 sentences. Decision: keep every token with train frequency >= 2
(14,309 types), plus one reserved `<UNK>` row (index 0) for anything not
seen at authoring time (e.g. a word introduced only in val/test, or any
true hapax). Frequency >= 2 rather than a fixed top-K: it is equally a
"top-K frequent tokens" rule per report 4.3, just expressed as a frequency
floor, and it is well-motivated on its own terms -- it keeps 96.6% of the
observed vocabulary (excluding only the 507 word-types seen exactly once,
which are dominated by typos, rare proper nouns, and one-off compounds not
worth a dedicated row) while avoiding a vocabulary size chosen to hit a
param-count target. `scripts/build_vocab.py` is the sole, deterministic,
mechanical producer of `vocab/vocab.json`, which is committed (it is a
frequency statistic over unlabeled text, not the dataset itself; the raw
sentence file remains gitignored).

## 2026-11-05T18:18Z -- Network architecture (fixed before any weight value)

Full design below is fixed now, before a single weight is written, so the
weight-authoring that follows is filling in a predetermined structure, not
shaping the structure around a result.

**Encoder** (`handnet/encode.py`, mechanical/structural only -- no sentiment
judgment lives here, only closed-class word-list membership and positional
logic that is identical regardless of what any word's sentiment is):

1. Tokenize (above).
2. Map each token to a vocab row (or UNK).
3. Clause split: scan for the first occurrence of a closed-class contrast
   marker (but/however/although/though/yet/while/despite/except/
   nonetheless/nevertheless/whereas/albeit). Tokens before it are the "pre"
   clause; the marker and everything after are the "post" clause (the
   marker token itself is excluded from word buckets but counted in a
   `has_contrast` flag). No contrast marker -> everything is "pre", "post"
   is empty. Only the first marker is used if more than one appears
   (documented simplification).
4. Negation scope: scanning left-to-right, a closed-class negator
   (not/n't/no/never/nobody/nothing/neither/nor/without/none/nowhere)
   opens a fixed forward window of SCOPE_WINDOW=4 tokens (counting every
   token, content or function, in that window); scope resets early on a
   clause boundary or on strong punctuation (. ! ? ; :). This is the
   standard negation-scope heuristic from the sentiment-analysis
   preprocessing literature (e.g. Pang & Lee 2002's "append _NEG within
   scope of negation"), applied uniformly and mechanically -- it is a
   structural rule about negator position, identical for every sentence,
   fixed before any weight exists, and computed over unlabeled text only.
5. Each non-function content token is assigned to exactly one of 3 buckets
   per clause -- plain / negated / modified -- modified meaning the token
   is immediately preceded by a closed-class degree-modifier word (see
   below) and not negated. Negators and contrast markers themselves are
   excluded from these buckets (they are function words, counted via flags
   instead).
6. Each clause x bucket (6 combinations) pools the embedding rows of its
   member tokens by SUM, then divides by sentence length (average, not sum
   pooling) so longer sentences do not automatically produce larger-
   magnitude scores purely from token count. This is a standard pooling
   normalization choice, not a sentiment judgment.
7. Encoder emits: 6 pooled 7-dim vectors (42 numbers) + 7 sentence-level
   structural scalars (has_contrast, pre_len_frac, post_len_frac,
   num_negators, has_exclaim, has_question, sentence_length) = 49 numbers.

**Per-word embedding table** `E`, shape (14310, 7). Columns, each an
independent, separately-justified semantic axis (rationale for *why these
7* axes, not yet the per-word values, which follow in later entries):

0. `valence` in [-3,3]: core sentiment polarity and strength, combined into
   one signed number (sign = direction, magnitude = strength) rather than
   two separate numbers, because I never need sign and magnitude addressed
   independently anywhere downstream -- only their product ever gets used.
1. `negation_sensitivity` in [0,1]: what fraction of a word's valence
   actually inverts under negation, vs. being merely neutralized. Motivated
   by this being genuinely non-uniform: "not good" reads as mildly-to-
   moderately negative (good negates fairly cleanly -- high sensitivity),
   but "not perfect" or "not amazing" reads as closer to lukewarm/neutral
   than to "terrible" (extreme/absolute words negate poorly -- low
   sensitivity). This is exactly the compositional phenomenon SST-2 was
   built to require (report section 3.1) and is not expressible as a
   single valence number alone. Set only where valence != 0.
2. `is_negator` in {0,1}: closed-class flag (the ~11 words listed above).
3. `is_contrast` in {0,1}: closed-class flag (the ~12 words listed above).
4. `degree_modifier` in [0, 2.5], 0 = not a degree word: unifies
   intensifiers and diminishers into one axis (>1 amplifies, <1 damps,
   0 = inert) since they are the same linguistic phenomenon at opposite
   ends of one scale, not two unrelated categories.
5. `certainty` in [-1,1]: hedge (seems/maybe/somewhat-as-hedge/apparently,
   negative) vs. assertive (clearly/definitely/undoubtedly, positive)
   marking. Deliberately separate from valence: "seems good" and "is good"
   differ in confidence, not in polarity, and SST-2's phrase-level
   compositional design (report 3.1) rewards exactly this kind of
   independent axis rather than folding everything into one sentiment
   number.
6. `content_negator` in {0,1}: content words whose own meaning is an
   absence/failure (lack, fails, missing, devoid, absence...) -- flagged
   separately from `valence` (which they also get, as ordinary negative
   content words) purely as an additional pooled signal the composition
   layer can use; no structural negation-propagation mechanism was built
   for this column in v1 (documented limitation, see the entry at the end
   of this section).

**Composition layer 1** (Linear(49 -> 12) + tanh), 12 named hidden units,
each hand-set to read only the handful of the 49 inputs it is semantically
about (the rest of that unit's weight row is 0): `plain_sentiment_pre`,
`negated_sentiment_pre`, `modified_sentiment_pre`, `plain_sentiment_post`,
`negated_sentiment_post`, `modified_sentiment_post`, `semantic_negation_
signal`, `certainty_signal`, `emphasis_signal`, `negation_density_signal`,
`question_discount_signal`, `pre_post_length_signal`.

**Composition layer 2 / fusion** (12 units + raw `has_contrast` skip input
-> 1 raw score): pre/post clause totals are combined with a contrast gate,
`pre_weight = 1 - has_contrast*(1-ALPHA)`, `post_weight = has_contrast*BETA`
-- when there is no "but", pre_weight=1 and post_weight=0 (post is empty
anyway); when there is, the post-clause is weighted to dominate
(ALPHA<1<BETA). This is the discourse-structure claim "X but Y" is
semantically dominated by Y, and it is implemented as plain arithmetic on a
0/1 structural flag (a gate, the same mechanism an LSTM/GRU forget-gate
uses), not a per-example branch.

**Output layer**: `logit = raw_score * output_scale + output_bias`,
`p = sigmoid(logit)`.

Total parameter budget: embedding 14310*7 = 100,170, composition layer 1
weights 49*12+12 = 600, fusion/output ~16 => ~100,786 total, matching the
brief's "order of 100k" target without padding -- every one of the 7
embedding columns and all 12 composition units is independently motivated
above, not sized to hit a number.

**Known v1 simplifications, stated honestly rather than hidden**: (a) only
the first contrast marker in a sentence is used for clause-splitting; (b)
negation scope is a fixed 4-token window for every negator rather than a
per-negator reach; (c) `content_negator` words do not structurally negate
a following word the way a real negator does, they only contribute an
extra pooled flag; (d) `degree_modifier` only looks one token back
(immediate adjacency), it does not handle "very, very good" stacking beyond
2 tokens specially (the second "very" simply also fires its own
degree_modifier contribution on the first "very" as if it were itself a
content token -- a minor mis-bucketing for stacked intensifiers, accepted
rather than special-cased).

## 2026-11-05T18:18Z -- Closed-class structural words (authoring/structural_words.yaml)

Checked every candidate function word against `vocab/vocab.json` first;
84 of 85 candidates are present at train frequency >= 2 (only "presumably"
is absent, so it is simply omitted -- it will fall back to UNK-like
all-zero treatment via no lexicon entry).

Non-obvious calls, logged individually per rule 5 of the brief:

- **"yet" excluded from `contrast_markers`** despite being a textbook
  contrast word, because SST-2 movie-review text frequently contains
  "not yet" (not yet rated / has n't been released yet). If "yet" also
  triggered a clause split, it would sever the negated word from its
  negator's scope in exactly the construction where getting negation
  right matters most. Kept as an ordinary nearly-neutral vocabulary word
  instead of a structural marker. This is a deliberate asymmetry: "but"
  essentially never co-occurs with "not" in a way that breaks negation
  scope, "yet" commonly does.
- **"while" kept** as a contrast marker despite being genuinely ambiguous
  between temporal ("while watching") and concessive ("while the lead is
  strong, the script is weak") senses, because the concessive sense
  dominates in critical-review register and a rare mis-split costs little
  (the pre/post split just lands in a slightly odd place, it does not
  flip a sign).
- **"too" treated as a plain amplifier** (degree_modifier=1.5), same
  family as "very", even though "too" carries an independent connotation
  of excess that can read as mildly negative regardless of the word it
  modifies ("too clever", "too long"). Capturing that would need "too" to
  carry its own small negative bias independent of what it modifies --
  a genuine extra mechanism, not just a weight value. Decided against
  adding a one-word special case; logged as a known gap rather than
  silently folded into a generic amplifier value.
- **"barely"/"hardly"/"scarcely" placed in `degree_modifiers` at 0.15**,
  not in `negators`, on the judgment that "barely enjoyable" still reads
  as weakly positive-leaning, not as negative as "not enjoyable" -- they
  dampen toward near-zero rather than flipping sign. This is a real,
  deliberate distinction from the `negators` list, not an oversight.
- **"kind"/"sort"/"bit" used as diminishers** ("kind of boring") at
  ~0.45-0.5, fully aware each also has an unrelated literal sense ("a kind
  person", "what sort of film", "a bit part") that a word-level (not a
  word-sense-level) model cannot distinguish. "kind" additionally gets a
  small positive `valence` in lexicon.yaml for its literal sense; the
  resulting sense-collision is accepted as a known limitation of
  bag-of-words-style encoding rather than hidden.
- **certainty is kept fully independent of valence and of
  degree_modifier**: "seems very good" should hedge a strong claim, and
  "clearly fine" should assert a weak one. Collapsing certainty into
  degree_modifier would make hedges act identically to diminishers, which
  they are not (a diminisher weakens the sentiment itself; a hedge
  weakens confidence in an otherwise-unweakened sentiment).
- **content_negators ("lack", "fail", "devoid"...) given no structural
  negation mechanism in v1** -- a real negator ("not") flips the bucket of
  a following content word; these words do not, they only set a pooled
  flag the composition layer can read as `semantic_negation_signal`. The
  alternative (treating "lacks" as a mini-negator that opens its own scope
  window) is a reasonable future extension but was deliberately deferred
  rather than rushed in: getting negator-opening logic right for a second,
  weaker class of trigger words adds real room for a wiring bug under time
  pressure, which is exactly the kind of mistake rule 4 (fix bugs, never
  retune values) is meant to make leaving-well-enough-alone the safer
  default. "miss"/"misses" included on the judgment that film-critique
  register favors the "fails to achieve/land" sense over "fondly long
  for"; logged as an accepted ambiguity rather than a confident claim.

## 2026-11-05T18:39Z -- Content-word lexicon (authoring/lexicon_families.py)

Authored per-word sentiment valence for content words as lemma families:
one judgment call per concept (e.g. "brilliant"), fanned out mechanically
to its real inflections ("brilliantly", "brilliance") that are actually
present in the vocabulary. A family is
`(forms, valence, sensitivity_or_None, note_or_None)`. `valence` is my own
judgment of polarity/strength in film-review register, anchored to a tier
scale I designed for internal consistency (+-3.0 superlative down to
+-0.2 bare) -- not copied from any external lexicon's scale (rule 3).
`sensitivity` encodes how cleanly "not <word>" negates the word: low for
extreme/absolute words ("not perfect" reads as merely imperfect, not bad),
high for plain gradable words ("not good" cleanly flips). Defaults to 0.7
when a family does not override it.

`scripts/expand_lexicon.py` mechanically expands every family against
`vocab/vocab.json`, flattens to one `word -> {valence, sensitivity}` dict,
and writes `authoring/lexicon.yaml`. The script carries zero judgment of
its own -- it only matches surface forms and fails loudly
(`sys.exit(1)`) if two families claim different valence for the same
word. This mirrors the existing `build_vocab.py` -> `vocab.json` split
between a hand-authored source and a mechanically-derived artifact.

**Authoring-time bugs caught before any run** (rule 4 allows fixing these;
none involved running the net or looking at any output):
- A bogus literal form `"fairly_as_adj"` (not a real word) in an early
  `fair` family, caught on review and fixed to `["fair"]` with a note
  distinguishing it from the unrelated `fairly` degree_modifier.
- `structural_words.yaml`'s `kind` comment promised a lexicon.yaml entry
  for the kind-hearted sense that had never actually been written; added
  `(["kind", "kindly", "kindness"], 0.6, 0.8, ...)` to fulfill it.
- Drafting batch 2 (below) left several dead placeholder tuples in the
  file with names like `"realistic_dup_removed"`, `"crass_dup_removed"`,
  `"suffering_as_verb"`, `"like_as_verb"` -- markers I used while
  consolidating duplicate-across-list notes into code, never cleaned up.
  Caught by grepping for `_dup_removed|_as_adj|_as_verb` before the first
  run on the new batch and removed; real forms that had been bundled into
  the same tuples (e.g. "suffered", "liked", "likes") were kept.

**First run** (batch 1, hand-recalled words without checking the
vocabulary first): 416 families, 675 matched forms, 675 distinct words,
258 unmatched forms, 28 zero-match families -- a ~72% hit rate against
the real vocabulary, since guessing inflected forms from memory without
checking which ones actually survive the frequency floor (`MIN_FREQ=2`
in `build_vocab.py`) is inherently lossy. The conflict check also caught
4 real authoring duplicates (same word, two different valences, from
independently-drafted tier lists: "triumphs/triumph", "glorious", a
"resplendent" merge, "touching") -- fixed by merging/removing the
redundant lines, not by changing any value a test or run had flagged.

675 distinct words is short of a comfortable "thousands of non-zero
weights" claim. The deficit is lexicon breadth (words I have not
authored yet), not vocabulary size -- `vocab.json` was already built
label-free from real training-sentence frequency, so shrinking the
frequency floor to manufacture more vocabulary entries would be an
unmotivated change in the wrong direction. Chose instead to find more
real words to author.

**Gap-check methodology (batch 2)**: computed
`vocab_words - (lexicon_words | structural_words)`, sorted by descending
train frequency (both sets and the frequency numbers were already
computed label-free, by code committed before this entry), and read
through the result by hand to find genuine sentiment-bearing words I had
missed. This is explicitly NOT output-driven tuning under rule 4: no
label is consulted, no model prediction is consulted, no model of any
kind runs at any point in this process. The only two inputs are (a) the
set of words already in my own authored files and (b) a raw unlabeled
word-frequency count -- both already-committed, non-label artifacts. The
check only tells me which real words exist to make a judgment call about;
it never tells me what judgment to make.

Reading that list surfaced ~250 genuine content words
(splendid, profound, irresistible, magical, admire, enthusiastic,
obnoxious, witless, soulless, paper-thin, phony, pointless, suck, crap,
skip, ... -- see `GAP_FILL_*` lists in lexicon_families.py) plus two gaps
in `structural_words.yaml`'s `degree_modifiers`: "extremely" was missing
entirely despite being a common strong intensifier, and "utter" (sibling
of the already-present "utterly") was also missing. While fixing those I
added four more intensifiers spotted the same way: "hugely", "wholly",
"insanely", "exceedingly", plus "overly" at the same amplify-only value
as "too" (rule: the degree_modifier mechanism amplifies, it does not
itself inject sentiment, even though "overly" nearly always collocates
with a complaint -- consistent with the existing "too" decision above).

**Deliberately excluded words** (same batch), because their polarity
flips with genre fit rather than with film quality, and a single
word-level weight would be more likely to mislead than help: scary,
frightening, terrifying, shocking, outrageous, dark, tension, guilty,
hot, fat, sick, complicated. A horror film being "scary" is praise; a
drama being "scary" (tonally) is more often a complaint -- no single
sign-and-magnitude weight represents both correctly, and the architecture
has no per-genre conditioning to disambiguate. Declining to author a
value here is a judgment call like any other, not a gap.

**Result after batch 2**: 651 families, 1001 matched forms, 1001 distinct
lexicon words (zero new conflicts, zero new zero-match families -- as
expected, since this batch was sourced directly from the vocabulary
rather than guessed). Combined with structural_words.yaml's 40
degree_modifiers + 15 certainty + 11 negators + 11 contrast_markers + 12
content_negators (7 of which deliberately collide with a lexicon.yaml
word, see structural_words.yaml notes), 1083 distinct words now carry at
least one hand-authored nonzero weight, for upward of 2000 nonzero
embedding-table entries alone once valence+sensitivity are both counted
-- comfortably clearing the brief's "thousands of non-zero weights"
target before the composition/fusion/output layers are even counted.

The 28 families with zero vocabulary matches (e.g. "unmatched",
"peerless", "tour-de-force", "catastrophe", "trainwreck") and the ~300
unmatched surface forms (mostly adverb/participle inflections like
"flawlessly", "mesmerizingly", "underachieved") are left as-is: they are
real words absent from the frequency-floor vocabulary, not authoring
errors, and adding them would require changing `vocab.json`'s frequency
floor -- out of scope for this pass and not motivated by any actual
deficit in the covered vocabulary.

## 2026-11-05T18:52Z -- content_negators lexicon gap (authoring/lexicon_families.py)

`structural_words.yaml`'s `content_negators` comment made the same kind
of promise the `kind` comment did (see the bug list above): it says
these words are "ordinary content words with their own valence entry in
lexicon.yaml" too, not just a structural flag. Checked and found that
promise only half-kept -- of the 12 content_negators (lack, lacks,
lacking, fail, fails, failing, missing, miss, misses, devoid, absence,
void), only `fail`/`fails`/`failing` actually had a lexicon.yaml entry
(-1.3, 0.7, from the pre-existing STRONG_NEGATIVE family). Added a new
`GAP_FILL_CONTENT_NEGATORS` family giving the other 9 their own valence:
`lack`/`lacks`/`lacking` at -1.0/0.6, `missing`/`miss`/`misses` at
-0.8/0.65 (judgment: the "fails to land/achieve" sense dominates in
film-critique register over "fondly long for" -- accepted ambiguity,
same call `structural_words.yaml` already documents for this word),
`devoid` at -1.1/0.5, `absence` at -0.6/0.6, `void` at -0.9/0.6 (the "void
of" sense). This is authoring new valence for words that already had
zero in lexicon.yaml, not retuning an existing value, so it is ordinary
authoring work, not a rule-4 post-test change.

`expand_lexicon.py`'s conflict check caught the obvious overlap: a first
draft of this family included `fail`/`fails`/`failing` at -1.2/0.6,
duplicating the pre-existing -1.3/0.7 STRONG_NEGATIVE entry with a
different value. Fixed by dropping the three duplicate forms from the
new family (left a comment pointing at the original), not by changing
either value -- the project's standing practice of never retuning an
already-authored value without cause.

**Re-run result**: 656 families, 1010 matched forms, 1010 distinct
lexicon words (up from 651/1001/1001), zero new conflicts, same 316
unmatched forms as before (the 9 unmatched forms this family's other
words would have added are forms I didn't draft, not new misses).

**Correction to the batch-2 entry above**: it wrote "12 content_negators
(7 of which deliberately collide with a lexicon.yaml word)". That
parenthetical conflated two different overlaps -- the real count of
content_negators with a lexicon valence at that time was 3
(fail/fails/failing), not 7; the other 4 were unrelated
degree_modifier/lexicon collisions (kind, amazingly, incredibly,
remarkably) that happen to sum to 7 when added to the 3. Recording the
correction here rather than editing that entry, since this log is
append-only and contemporaneous by design. All 12 content_negators now
have a lexicon valence, so the distinction is moot going forward:
recomputed directly (not by hand-arithmetic, to avoid repeating the same
kind of error) -- lexicon.yaml now has 1010 distinct words,
structural_words.yaml's five weighted/flagged lists (negators,
contrast_markers, degree_modifiers, certainty, content_negators) have 89
distinct words, 16 of those also carry a lexicon valence, so 1083
distinct words carry at least one hand-authored nonzero weight -- the
same combined total as before by coincidence (each newly-dual-purpose
word was already counted on the structural side).
