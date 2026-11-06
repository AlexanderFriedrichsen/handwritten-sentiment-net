"""Names and index layout for composition layer 1's 12 hidden units and
the encoder's 49-number output.

Fixed by the architecture design in authoring/RATIONALE.md (the
"Composition/fusion/output design" entry) before any weight value
existed -- this is mechanism, not a judgment call, so it lives in code
rather than in an authoring/*.yaml file. authoring/composition_weights.yaml's
header comments mirror this layout; if the two ever disagree, this module
is what the code actually runs on.
"""

from handnet.columns import N_EMBED_COLS

UNIT_NAMES = (
    "plain_sentiment_pre",
    "negated_sentiment_pre",
    "modified_sentiment_pre",
    "plain_sentiment_post",
    "negated_sentiment_post",
    "modified_sentiment_post",
    "semantic_negation_signal",
    "certainty_signal",
    "emphasis_signal",
    "negation_density_signal",
    "question_discount_signal",
    "pre_post_length_signal",
)
N_UNITS = len(UNIT_NAMES)

# Encoder output layout: 6 pooled blocks x 7 embedding columns (42) + 7
# sentence-level scalars = 49. Block order and scalar order fixed in
# RATIONALE.md's "Composition/fusion/output design" entry.
BLOCK_NAMES = (
    "pre_plain",
    "pre_negated",
    "pre_modified",
    "post_plain",
    "post_negated",
    "post_modified",
)
N_BLOCKS = len(BLOCK_NAMES)
COLS_PER_BLOCK = N_EMBED_COLS

SCALAR_NAMES = (
    "has_contrast",
    "pre_len_frac",
    "post_len_frac",
    "num_negators",
    "has_exclaim",
    "has_question",
    "sentence_length",
)
N_SCALARS = len(SCALAR_NAMES)

N_ENCODER_OUTPUTS = N_BLOCKS * COLS_PER_BLOCK + N_SCALARS
assert N_ENCODER_OUTPUTS == 49

BLOCK_BASE = {name: i * COLS_PER_BLOCK for i, name in enumerate(BLOCK_NAMES)}
SCALAR_BASE = N_BLOCKS * COLS_PER_BLOCK
SCALAR_INDEX = {name: SCALAR_BASE + i for i, name in enumerate(SCALAR_NAMES)}

# Absolute indices that are always exactly 0 in every pooled vector for
# every sentence (the is_negator/is_contrast column of each of the 6
# blocks) -- negators/contrast markers are structurally excluded from all
# buckets. No composition unit should assign nonzero weight to these.
from handnet.columns import IS_CONTRAST, IS_NEGATOR  # noqa: E402

DEAD_INDICES = tuple(
    sorted(
        BLOCK_BASE[name] + col
        for name in BLOCK_NAMES
        for col in (IS_NEGATOR, IS_CONTRAST)
    )
)
assert DEAD_INDICES == (2, 3, 9, 10, 16, 17, 23, 24, 30, 31, 37, 38)
