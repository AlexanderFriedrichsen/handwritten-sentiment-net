"""Column indices for the per-word embedding table E.

Fixed by the architecture design in authoring/RATIONALE.md (the
"Network architecture" entry) before any weight value existed -- this
is mechanism, not a judgment call, so it lives in code rather than in an
authoring/*.yaml file.
"""

VALENCE = 0
NEGATION_SENSITIVITY = 1
IS_NEGATOR = 2
IS_CONTRAST = 3
DEGREE_MODIFIER = 4
CERTAINTY = 5
CONTENT_NEGATOR = 6

N_EMBED_COLS = 7
