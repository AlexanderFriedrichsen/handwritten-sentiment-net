"""Hand-authored sentiment valence judgments, by lemma family.

This is the primary source of my semantic judgment for content words (as
opposed to the closed-class function words in structural_words.yaml).
Every value below is my own assessment of the word's polarity and strength
in film-review register, formed by reasoning about the word directly --
none of it is copied or derived from any sentiment lexicon, dataset
statistic, or label. See authoring/RATIONALE.md for dated entries
discussing the non-obvious calls.

A family groups surface forms that share one polarity judgment (e.g. an
adjective and its adverb: "brilliant"/"brilliantly") so the judgment is
recorded once per concept, not once per inflection. `scripts/expand_
lexicon.py` deterministically expands each family against the frozen
vocabulary (vocab/vocab.json) and writes the flat authoring/lexicon.yaml
that scripts/compile_weights.py actually reads -- the flat file is a
mechanical derivation of this file, not an independent source of judgment.

Tuple shape: (forms, valence, sensitivity_or_None, note_or_None)
  - forms: list[str] of surface forms (only those present in the frequency
    vocabulary end up with a weight; absent forms are reported, not an
    error).
  - valence: float in [-3, 3]. Anchor tiers I used while authoring
    (not a formula, a consistency aid): +-3.0 superlative, +-2.5 very
    strong, +-2.0 strong, +-1.5 solid, +-1.0 plain, +-0.6 mild, +-0.3 bare.
  - sensitivity: float in [0, 1] or None to use the global default (0.7,
    see RATIONALE.md). Lower = negates poorly (extreme/absolute words);
    higher = negates cleanly (plain gradable words).
  - note: short rationale for anything non-obvious; None when the call is
    self-evident from the tier alone.
"""

DEFAULT_SENSITIVITY = 0.7

# ---------------------------------------------------------------------------
# SUPERLATIVE POSITIVE (extreme/absolute -- negates poorly, low sensitivity)
# ---------------------------------------------------------------------------
SUPERLATIVE_POSITIVE = [
    (["masterpiece", "masterpieces"], 3.0, 0.25, "'not a masterpiece' is still far from bad"),
    (["perfection"], 3.0, 0.25, None),
    (["flawless", "flawlessly"], 2.9, 0.25, None),
    (["immaculate", "immaculately"], 2.8, 0.3, None),
    (["impeccable", "impeccably"], 2.8, 0.3, None),
    (["unparalleled"], 2.8, 0.3, None),
    (["unmatched"], 2.6, 0.3, None),
    (["unsurpassed"], 2.7, 0.3, None),
    (["peerless"], 2.7, 0.3, None),
    (["transcendent", "transcendently"], 2.7, 0.3, None),
    (["sublime", "sublimely"], 2.7, 0.3, None),
    (["perfect", "perfectly"], 2.6, 0.3, "'not perfect' reads as merely imperfect, not bad"),
]

# ---------------------------------------------------------------------------
# VERY STRONG POSITIVE
# ---------------------------------------------------------------------------
VERY_STRONG_POSITIVE = [
    (["brilliant", "brilliantly", "brilliance"], 2.6, 0.3, None),
    (["superb", "superbly"], 2.6, 0.35, None),
    (["extraordinary", "extraordinarily"], 2.5, 0.3, None),
    (["magnificent", "magnificently"], 2.6, 0.3, None),
    (["exquisite", "exquisitely"], 2.5, 0.35, None),
    (["dazzling", "dazzlingly", "dazzles", "dazzled", "dazzle"], 2.4, 0.4, None),
    (["breathtaking", "breathtakingly"], 2.5, 0.35, None),
    (["phenomenal", "phenomenally"], 2.5, 0.3, None),
    (["spectacular", "spectacularly"], 2.4, 0.4, None),
    (["triumphant", "triumphantly", "triumph", "triumphs", "triumphed"], 2.3, 0.4, None),
    (["glorious", "gloriously", "resplendent"], 2.3, 0.4, None),
    (["luminous", "luminously"], 2.2, 0.45, "film-critic register: often praises a performance's glow"),
    (["radiant", "radiantly"], 2.1, 0.45, None),
    (["masterful", "masterfully"], 2.5, 0.35, None),
    (["virtuoso", "virtuosic"], 2.4, 0.4, None),
    (["genius"], 2.5, 0.35, None),
    (["mesmerizing", "mesmerizingly", "mesmerizes", "mesmerized", "mesmerize"], 2.3, 0.45, None),
    (["electrifying", "electrified", "electrifies"], 2.3, 0.45, None),
    (["staggering", "staggeringly"], 2.2, 0.45, None),
    (["astonishing", "astonishingly", "astonishes", "astonished", "astonish"], 2.2, 0.45, None),
    (["astounding", "astoundingly", "astounds", "astounded"], 2.2, 0.45, None),
    (["tour-de-force"], 2.6, 0.3, None),
    (["bravura"], 2.4, 0.4, None),
]

# ---------------------------------------------------------------------------
# STRONG POSITIVE
# ---------------------------------------------------------------------------
STRONG_POSITIVE = [
    (["excellent", "excellently", "excels", "excel", "excelled", "excelling"], 2.1, 0.5, None),
    (["outstanding", "outstandingly"], 2.1, 0.5, None),
    (["wonderful", "wonderfully"], 2.0, 0.55, None),
    (["marvelous", "marvelously"], 2.0, 0.5, None),
    (["remarkable", "remarkably"], 2.0, 0.5, None),
    (["stunning", "stunningly", "stuns", "stunned"], 2.1, 0.5, None),
    (["incredible", "incredibly"], 2.0, 0.5, "note: also used as a plain intensifier elsewhere; here as a predicate of praise"),
    (["amazing", "amazingly", "amazes", "amazed", "amaze"], 2.0, 0.5, None),
    (["captivating", "captivatingly", "captivates", "captivated", "captivate"], 2.0, 0.55, None),
    (["riveting", "rivets", "riveted", "rivet"], 2.0, 0.55, None),
    (["gripping", "grips", "gripped", "grip"], 1.9, 0.55, None),
    (["exhilarating", "exhilaratingly", "exhilarated", "exhilarates", "exhilarate"], 2.0, 0.5, None),
    (["thrilling", "thrillingly", "thrills", "thrilled", "thrill"], 1.9, 0.55, None),
    (["powerful", "powerfully"], 1.8, 0.6, None),
    (["unforgettable"], 2.1, 0.45, None),
    (["timeless", "timelessly"], 1.9, 0.5, None),
    (["exceptional", "exceptionally"], 2.1, 0.45, None),
    (["superlative"], 2.1, 0.45, None),
    (["inspired"], 1.8, 0.55, "as praise ('an inspired choice'), not the literal passive verb"),
    (["towering"], 1.9, 0.5, None),
    (["exemplary"], 2.0, 0.45, None),
    (["soars", "soared", "soaring", "soar"], 1.9, 0.55, None),
    (["euphoric", "euphorically"], 1.9, 0.5, None),
]

# ---------------------------------------------------------------------------
# SOLID POSITIVE
# ---------------------------------------------------------------------------
SOLID_POSITIVE = [
    (["great", "greatly"], 1.6, 0.65, None),
    (["impressive", "impressively", "impresses", "impressed", "impress"], 1.6, 0.6, None),
    (["compelling", "compellingly"], 1.7, 0.6, None),
    (["engaging", "engagingly", "engages", "engaged", "engage"], 1.6, 0.65, None),
    (["charming", "charmingly", "charms", "charmed", "charm"], 1.6, 0.65, None),
    (["delightful", "delightfully", "delights", "delighted", "delight"], 1.6, 0.6, None),
    (["enchanting", "enchantingly", "enchants", "enchanted", "enchant"], 1.7, 0.6, None),
    (["poignant", "poignantly", "poignancy"], 1.6, 0.6, None),
    (["moving"], 1.5, 0.65, "emotional sense only; bare verb 'move/moves' kept neutral, see RATIONALE.md"),
    (["touching"], 1.4, 0.65, "emotional sense only; bare verb 'touch/touches' kept neutral"),
    (["heartfelt"], 1.5, 0.6, None),
    (["heartwarming"], 1.6, 0.55, None),
    (["funny", "funnier", "funniest"], 1.5, 0.7, None),
    (["hilarious", "hilariously"], 1.8, 0.55, None),
    (["witty", "wittily", "wit"], 1.6, 0.6, None),
    (["clever", "cleverly", "cleverness"], 1.5, 0.65, None),
    (["sharp", "sharply"], 1.3, 0.65, "wit/intelligence sense"),
    (["insightful", "insightfully"], 1.6, 0.6, None),
    (["nuanced"], 1.7, 0.55, None),
    (["assured", "assuredly"], 1.4, 0.6, "confident-craft sense"),
    (["accomplished"], 1.5, 0.6, None),
    (["polished"], 1.4, 0.65, None),
    (["vivid", "vividly"], 1.4, 0.65, None),
    (["lush", "lushly"], 1.3, 0.65, None),
    (["rich", "richly", "richness"], 1.3, 0.65, None),
    (["textured"], 1.2, 0.65, None),
    (["admirable", "admirably"], 1.5, 0.6, None),
    (["inventive", "inventively", "inventiveness"], 1.5, 0.6, None),
    (["imaginative", "imaginatively"], 1.5, 0.6, None),
    (["creative", "creatively", "creativity"], 1.4, 0.65, None),
    (["innovative", "innovatively"], 1.5, 0.6, None),
    (["groundbreaking"], 1.7, 0.55, None),
    (["bold", "boldly", "boldness"], 1.3, 0.65, None),
    (["daring", "daringly"], 1.3, 0.65, None),
    (["ambitious", "ambitiously"], 1.2, 0.7, "double-edged: often paired with a caveat; kept solidly positive on its own"),
    (["confident", "confidently"], 1.2, 0.7, None),
    (["fresh", "freshness"], 1.3, 0.65, None),
    (["original", "originality"], 1.4, 0.65, None),
    (["elegant", "elegantly", "elegance"], 1.4, 0.65, None),
    (["graceful", "gracefully"], 1.3, 0.65, None),
    (["beautiful", "beautifully"], 1.6, 0.6, None),
    (["gorgeous", "gorgeously"], 1.6, 0.6, None),
    (["haunting", "hauntingly"], 1.5, 0.6, "critic-praise sense: memorably evocative, not frightening"),
    (["arresting"], 1.6, 0.6, None),
    (["shines", "shone", "shining", "shine"], 1.5, 0.6, None),
    (["sparkles", "sparkled", "sparkling", "sparkle"], 1.5, 0.6, None),
    (["crackles", "crackled", "crackling", "crackle"], 1.4, 0.6, "'dialogue crackles with wit'"),
    (["sings"], 1.4, 0.6, "metaphorical praise, 'the dialogue sings'"),
    (["flourishes", "flourished", "flourishing", "flourish"], 1.4, 0.6, None),
    (["succeeds", "succeeded", "succeeding", "succeed", "successful", "successfully"], 1.4, 0.65, None),
    (["uplifting"], 1.6, 0.55, None),
    (["cathartic", "catharsis"], 1.5, 0.55, None),
    (["life-affirming"], 1.7, 0.5, None),
    (["stirring", "stirringly"], 1.5, 0.6, None),
    (["rousing", "rousingly"], 1.5, 0.6, None),
    (["joyous", "joyful", "joyfully"], 1.6, 0.6, None),
    (["gut-wrenching"], 1.5, 0.55, "praise for emotional force, not literally negative"),
    (["devastating", "devastatingly"], 1.3, 0.55, "as critical praise for impact, e.g. 'a devastating performance'"),
    (["harrowing"], 1.2, 0.55, "ambiguous: can read as praise (intense/affecting) or complaint (grueling); leaned mild positive, see RATIONALE.md"),
]

# ---------------------------------------------------------------------------
# PLAIN POSITIVE (clean gradable words -- negate well, high sensitivity)
# ---------------------------------------------------------------------------
PLAIN_POSITIVE = [
    (["good", "well"], 1.0, 0.9, None),
    (["enjoyable", "enjoyably", "enjoys", "enjoyed", "enjoy", "enjoying"], 1.0, 0.85, None),
    (["likable", "likeable"], 0.9, 0.85, None),
    (["appealing", "appealingly"], 0.9, 0.8, None),
    (["solid", "solidly"], 0.9, 0.8, None),
    (["fine"], 0.7, 0.85, None),
    (["satisfying", "satisfyingly", "satisfies", "satisfied", "satisfy"], 1.0, 0.8, None),
    (["rewarding", "rewardingly"], 1.0, 0.75, None),
    (["entertaining", "entertainingly", "entertains", "entertained", "entertain"], 1.0, 0.8, None),
    (["fun"], 1.0, 0.85, None),
    (["warm", "warmly", "warmth"], 0.9, 0.8, None),
    (["sweet", "sweetly", "sweetness"], 0.8, 0.8, None),
    (["endearing", "endearingly"], 1.0, 0.75, None),
    (["earnest", "earnestly", "earnestness"], 0.7, 0.8, None),
    (["sincere", "sincerely", "sincerity"], 0.8, 0.8, None),
    (["honest", "honestly", "honesty"], 0.7, 0.8, None),
    (["thoughtful", "thoughtfully"], 1.0, 0.75, None),
    (["smart", "smartly"], 1.0, 0.8, None),
    (["effective", "effectively"], 0.9, 0.8, None),
    (["skillful", "skillfully", "skilled", "skill"], 1.0, 0.75, None),
    (["deft", "deftly"], 1.0, 0.75, None),
    (["lovely", "loveliness"], 1.0, 0.8, None),
    (["pleasing", "pleasingly", "pleased", "pleases", "please"], 0.9, 0.8, None),
    (["winning"], 1.0, 0.75, "praise sense ('a winning performance')"),
    (["affecting"], 1.0, 0.75, None),
    (["resonant", "resonates", "resonated", "resonate", "resonance"], 1.0, 0.75, None),
    (["memorable", "memorably"], 1.1, 0.7, None),
    (["vibrant", "vibrantly", "vibrancy"], 1.1, 0.75, None),
    (["energetic", "energetically", "energy"], 0.9, 0.8, None),
    (["spirited", "spiritedly"], 0.9, 0.8, None),
    (["lively", "liveliness"], 0.9, 0.8, None),
    (["dynamic", "dynamically"], 0.8, 0.8, None),
    (["well-crafted", "well-made", "well-acted", "well-written", "well-directed", "well-paced", "well-observed"], 1.3, 0.7, None),
    (["natural", "naturally"], 0.6, 0.8, "unforced-acting sense"),
    (["authentic", "authentically", "authenticity"], 1.0, 0.75, None),
    (["believable", "believably"], 0.9, 0.8, None),
    (["convincing", "convincingly"], 0.9, 0.8, None),
    (["charismatic", "charisma"], 1.3, 0.7, None),
    (["magnetic"], 1.2, 0.7, None),
    (["sympathetic"], 0.9, 0.75, None),
    (["relatable"], 0.9, 0.75, None),
    (["layered"], 1.0, 0.75, None),
    (["complex"], 0.7, 0.8, "character-complexity sense, usually positive"),
    (["multidimensional", "multi-dimensional", "three-dimensional"], 1.0, 0.75, None),
    (["brisk", "briskly"], 0.7, 0.8, "pacing praise"),
    (["tight", "tightly"], 0.8, 0.8, "pacing/editing praise"),
    (["taut", "tautly"], 1.0, 0.75, None),
    (["snappy", "snappily"], 0.9, 0.8, None),
    (["effortless", "effortlessly"], 1.1, 0.7, None),
    (["cute", "cutely"], 0.6, 0.8, None),
    (["amusing", "amusingly"], 0.9, 0.8, None),
    (["comic"], 0.6, 0.8, None),
    (["laugh-out-loud"], 1.3, 0.65, None),
    (["heartening", "heartens"], 1.1, 0.7, None),
    (["gritty", "grittiness"], 0.5, 0.8, "authenticity-of-tone sense, usually praise"),
    (["raw"], 0.4, 0.8, "visceral-authenticity sense; context-dependent"),
    (["visceral", "viscerally"], 0.9, 0.75, None),
    (["subtle", "subtly", "subtlety"], 1.1, 0.7, None),
    (["understated", "understatedly"], 1.0, 0.75, None),
    (["restrained", "restraint"], 0.7, 0.75, "praise-for-control sense; see RATIONALE.md for the ambiguity"),
]

# ---------------------------------------------------------------------------
# MILD / BARE POSITIVE
# ---------------------------------------------------------------------------
MILD_POSITIVE = [
    (["decent", "decently"], 0.7, 0.8, None),
    (["respectable", "respectably"], 0.6, 0.8, None),
    (["competent", "competently", "competence"], 0.6, 0.8, None),
    (["watchable"], 0.5, 0.8, None),
    (["pleasant", "pleasantly"], 0.6, 0.8, None),
    (["agreeable", "agreeably"], 0.5, 0.8, None),
    (["interesting", "interestingly"], 0.4, 0.8, "frequently a lukewarm/backhanded compliment in film criticism"),
    (["intriguing", "intriguingly"], 0.7, 0.75, None),
    (["curious", "curiously"], 0.2, 0.8, "often near-neutral descriptive rather than evaluative"),
    (["promising"], 0.6, 0.75, "often paired with an unmet payoff; kept modest"),
    (["modest", "modestly"], 0.4, 0.8, None),
    (["amiable", "amiably"], 0.5, 0.8, None),
    (["genial", "genially"], 0.5, 0.8, None),
    (["affable", "affably"], 0.5, 0.8, None),
    (["worthwhile"], 0.7, 0.8, None),
    (["commendable", "commendably"], 0.7, 0.75, None),
    (["praiseworthy"], 0.8, 0.75, None),
    (["tidy", "tidily"], 0.4, 0.8, None),
    (["efficient", "efficiently"], 0.4, 0.8, None),
    (["serviceable"], 0.3, 0.8, None),
    (["okay", "ok"], 0.3, 0.85, None),
    (["adequate", "adequately"], 0.3, 0.8, None),
    (["passable"], 0.2, 0.8, None),
    (["fair"], 0.2, 0.8, "adjective sense ('a fair attempt'); 'fairly' is already a degree_modifier, not duplicated here"),
    (["reasonable", "reasonably"], 0.3, 0.8, None),
    (["harmless", "harmlessly"], 0.1, 0.8, None),
    (["inoffensive"], 0.1, 0.8, None),
    (["unassuming"], 0.3, 0.8, None),
    (["gentle", "gently"], 0.4, 0.8, None),
    (["kind", "kindly", "kindness"], 0.6, 0.8, "kind-hearted sense; 'kind' the degree_modifier ('kind of') is a separate colliding sense in structural_words.yaml"),
]

# ---------------------------------------------------------------------------
# BARE / MILD NEGATIVE
# ---------------------------------------------------------------------------
MILD_NEGATIVE = [
    (["uneven", "unevenly"], -0.8, 0.8, None),
    (["flawed"], -0.8, 0.8, None),
    (["forgettable"], -0.9, 0.75, None),
    (["unremarkable"], -0.8, 0.8, None),
    (["unmemorable"], -0.9, 0.75, None),
    (["mediocre", "mediocrity"], -1.0, 0.8, None),
    (["average"], -0.5, 0.8, "below-expectations sense in a review context, not purely neutral"),
    (["middling"], -0.8, 0.8, None),
    (["so-so"], -0.7, 0.8, None),
    (["lackluster"], -1.1, 0.75, None),
    (["insipid", "insipidly"], -1.2, 0.7, None),
    (["vapid", "vapidly"], -1.2, 0.7, None),
    (["disappointing", "disappointingly", "disappoints", "disappointed", "disappoint"], -1.1, 0.8, None),
    (["underwhelming", "underwhelmingly", "underwhelms", "underwhelmed"], -1.1, 0.75, None),
    (["patchy"], -0.7, 0.8, None),
    (["inconsistent", "inconsistently", "inconsistency"], -0.7, 0.8, None),
    (["minor"], -0.2, 0.8, None),
    (["negligible"], -0.3, 0.8, None),
    (["questionable", "questionably"], -0.4, 0.8, None),
    (["dubious", "dubiously"], -0.5, 0.8, None),
    (["shaky", "shakily"], -0.6, 0.8, None),
    (["wobbly"], -0.6, 0.8, None),
    (["iffy"], -0.5, 0.8, None),
    (["slight"], -0.2, 0.8, "minor-flaw sense; the degree_modifier 'slightly' is handled separately"),
    (["thin", "thinly"], -0.8, 0.8, "insubstantial-content sense"),
    (["unsettling", "unsettled"], -0.2, 0.8, "ambiguous, can be deliberate/positive in genre film; leaned mild negative"),
    (["depressing", "depressingly"], -0.5, 0.8, "ambiguous (can be praise-for-impact); default leaned mild negative"),
    (["bleak", "bleakness"], -0.3, 0.8, "often tone-descriptive rather than evaluative"),
    (["grim", "grimly"], -0.4, 0.8, None),
    (["disturbing", "disturbingly"], -0.2, 0.8, "ambiguous, often deliberate in the genre; default mild negative"),
    (["campy", "camp"], -0.2, 0.8, "ambiguous (can be affectionate); default mild negative"),
    (["over-the-top"], -0.5, 0.8, None),
]

# ---------------------------------------------------------------------------
# PLAIN NEGATIVE (clean gradable words)
# ---------------------------------------------------------------------------
PLAIN_NEGATIVE = [
    (["bad", "badly"], -1.0, 0.9, None),
    (["boring", "boringly", "bores", "bored", "bore"], -1.3, 0.85, None),
    (["dull", "dully", "dullness"], -1.2, 0.85, None),
    (["flat", "flatly", "flatness"], -1.1, 0.8, "pacing/affect sense"),
    (["bland", "blandly", "blandness"], -1.1, 0.8, None),
    (["stale", "staleness"], -1.1, 0.75, None),
    (["trite", "tritely"], -1.1, 0.75, None),
    (["predictable", "predictably", "predictability"], -1.0, 0.8, None),
    (["weak", "weakly", "weakness"], -0.9, 0.8, None),
    (["flimsy", "flimsily"], -1.0, 0.75, None),
    (["slow", "slowly", "slowness"], -0.6, 0.8, "pacing sense; mild since not always a flaw"),
    (["dreary", "drearily", "dreariness"], -1.2, 0.75, None),
    (["dry", "dryly", "dryness"], -0.6, 0.8, "can be deliberate wit or genuine flaw; leaned mild-plain negative"),
    (["monotonous", "monotonously", "monotony"], -1.2, 0.75, None),
    (["tiresome", "tiresomely"], -1.3, 0.75, None),
    (["repetitive", "repetitively", "repetition"], -1.0, 0.8, None),
    (["uninteresting"], -1.1, 0.8, None),
    (["uninspiring", "uninspired"], -1.1, 0.75, None),
    (["unimpressive"], -1.0, 0.8, None),
    (["unconvincing", "unconvincingly"], -1.1, 0.8, None),
    (["confusing", "confusingly", "confuses", "confused", "confuse"], -0.9, 0.8, None),
    (["annoying", "annoyingly", "annoys", "annoyed", "annoy"], -1.2, 0.8, None),
    (["irritating", "irritatingly", "irritates", "irritated", "irritate"], -1.2, 0.8, None),
    (["grating", "gratingly"], -1.3, 0.75, None),
    (["tiring", "tires", "tired", "tire"], -0.9, 0.8, None),
    (["exhausting", "exhausts", "exhausted", "exhaust"], -1.0, 0.75, None),
    (["unlikable", "unlikeable"], -1.1, 0.8, None),
    (["unsympathetic"], -1.0, 0.8, None),
    (["charmless"], -1.2, 0.75, None),
    (["wooden", "woodenly"], -1.2, 0.75, "acting-stiffness sense"),
    (["stiff", "stiffly", "stiffness"], -0.9, 0.8, "acting/pacing sense"),
    (["stilted", "stiltedly"], -1.2, 0.75, None),
    (["lifeless", "listless", "listlessly"], -1.2, 0.75, None),
    (["sluggish", "sluggishly"], -1.1, 0.75, None),
    (["plodding", "plods", "plod"], -1.2, 0.75, None),
    (["draggy", "drags", "dragged", "drag"], -1.1, 0.75, None),
    (["overlong"], -1.0, 0.75, None),
    (["bloated"], -1.1, 0.75, None),
    (["padded", "padding"], -0.9, 0.8, None),
    (["meandering", "meanders", "meandered", "meander"], -1.0, 0.75, None),
    (["glacial", "glacially"], -1.1, 0.75, "pacing sense"),
    (["sags", "sagged", "sagging", "sag"], -0.9, 0.8, "pacing sense ('the middle sags')"),
    (["falters", "faltered", "faltering", "falter"], -1.0, 0.75, None),
    (["stumbles", "stumbled", "stumbling", "stumble"], -0.9, 0.75, None),
    (["flounders", "floundered", "floundering", "flounder"], -1.1, 0.75, None),
    (["misfires", "misfired", "misfiring", "misfire"], -1.2, 0.7, None),
    (["fizzles", "fizzled", "fizzling", "fizzle"], -1.1, 0.75, None),
    (["sputters", "sputtered", "sputtering", "sputter"], -1.0, 0.75, None),
    (["squanders", "squandered", "squandering", "squander"], -1.3, 0.7, None),
    (["wastes", "wasted", "wasting", "waste"], -1.1, 0.75, "'wastes a talented cast'"),
    (["cardboard"], -1.3, 0.7, "flat-characterization sense"),
    (["cartoonish", "cartoonishly"], -1.0, 0.75, None),
    (["caricature", "caricatures"], -0.9, 0.75, None),
    (["stereotype", "stereotypical", "stereotypes", "stereotyped"], -1.1, 0.75, None),
    (["stock"], -0.7, 0.8, "'stock characters'"),
    (["one-dimensional"], -1.3, 0.7, None),
    (["unfunny"], -1.3, 0.75, None),
    (["humorless", "humorlessly"], -1.1, 0.75, None),
]

# ---------------------------------------------------------------------------
# SOLID NEGATIVE
# ---------------------------------------------------------------------------
SOLID_NEGATIVE = [
    (["tedious", "tediously", "tedium"], -1.6, 0.65, None),
    (["clichéd", "cliched", "cliche", "cliché", "cliches", "clichés"], -1.5, 0.65, None),
    (["hackneyed"], -1.6, 0.6, None),
    (["formulaic", "formulaically"], -1.5, 0.65, None),
    (["contrived", "contrivance", "contrivances"], -1.5, 0.65, None),
    (["derivative"], -1.4, 0.65, None),
    (["unoriginal", "unoriginality"], -1.4, 0.65, None),
    (["hollow", "hollowly", "hollowness"], -1.4, 0.65, None),
    (["shallow", "shallowly", "shallowness"], -1.3, 0.65, None),
    (["superficial", "superficially", "superficiality"], -1.3, 0.65, None),
    (["forced", "forcedly"], -1.1, 0.7, "unnatural-effort sense"),
    (["strained", "strainedly"], -1.1, 0.7, None),
    (["labored", "laboured"], -1.2, 0.65, None),
    (["clumsy", "clumsily", "clumsiness"], -1.2, 0.7, None),
    (["awkward", "awkwardly", "awkwardness"], -1.0, 0.7, None),
    (["disjointed", "disjointedly"], -1.3, 0.65, None),
    (["incoherent", "incoherently", "incoherence"], -1.5, 0.6, None),
    (["muddled"], -1.3, 0.65, None),
    (["messy", "messily"], -1.1, 0.7, None),
    (["sloppy", "sloppily", "sloppiness"], -1.3, 0.65, None),
    (["careless", "carelessly", "carelessness"], -1.1, 0.7, None),
    (["lazy", "lazily", "laziness"], -1.2, 0.65, "craft-effort sense"),
    (["unimaginative"], -1.2, 0.65, None),
    (["generic", "generically"], -1.1, 0.7, None),
    (["heavy-handed", "heavyhanded"], -1.3, 0.65, None),
    (["ham-fisted"], -1.3, 0.65, None),
    (["preachy", "preachiness"], -1.3, 0.65, None),
    (["sanctimonious", "sanctimoniously"], -1.4, 0.6, None),
    (["pretentious", "pretentiously", "pretension", "pretentiousness"], -1.4, 0.65, None),
    (["self-indulgent", "indulgent", "indulgently"], -1.3, 0.65, None),
    (["overwrought"], -1.4, 0.65, None),
    (["overblown"], -1.3, 0.65, None),
    (["overdone"], -1.2, 0.65, None),
    (["excessive", "excessively"], -1.0, 0.7, None),
    (["gratuitous", "gratuitously"], -1.3, 0.65, None),
    (["exploitative"], -1.3, 0.6, None),
    (["cheap", "cheaply", "cheapness"], -1.1, 0.7, "craft-quality sense, not literal price"),
    (["shoddy", "shoddily"], -1.4, 0.6, None),
    (["cheesy", "cheesiness"], -1.0, 0.7, None),
    (["corny", "corniness"], -1.0, 0.7, None),
    (["hokey", "hokiness"], -1.1, 0.65, None),
    (["silly", "silliness"], -0.9, 0.75, "default critical sense; can be affectionate in context, see RATIONALE.md"),
    (["ridiculous", "ridiculously", "ridiculousness"], -1.3, 0.65, None),
    (["absurd", "absurdly", "absurdity"], -1.1, 0.65, "can be deliberate/comic; default negative"),
    (["implausible", "implausibly", "implausibility"], -1.2, 0.65, None),
    (["far-fetched"], -1.2, 0.65, None),
    (["unbelievable", "unbelievably"], -1.1, 0.7, "credibility sense, distinct from the awe sense of 'incredible'"),
    (["gimmicky", "gimmick", "gimmicks"], -1.2, 0.65, None),
    (["self-important", "self-importance"], -1.3, 0.6, None),
    (["navel-gazing"], -1.2, 0.6, None),
    (["overacted", "overacting", "overact", "overacts"], -1.4, 0.6, None),
    (["hammy", "ham"], -1.2, 0.65, None),
    (["miscast"], -1.3, 0.6, None),
    (["underwritten"], -1.3, 0.6, None),
    (["underdeveloped"], -1.2, 0.65, None),
    (["schmaltzy", "schmaltz"], -1.2, 0.65, None),
    (["maudlin"], -1.3, 0.6, None),
    (["sappy", "sappiness"], -1.1, 0.65, None),
    (["saccharine"], -1.3, 0.6, None),
    (["interminable", "interminably"], -1.6, 0.6, "pacing sense"),
    (["fails", "failed", "failing", "fail"], -1.3, 0.7, None),
    (["underachieves", "underachieved", "underachieve"], -1.1, 0.7, None),
]

# ---------------------------------------------------------------------------
# STRONG NEGATIVE
# ---------------------------------------------------------------------------
STRONG_NEGATIVE = [
    (["terrible", "terribly"], -2.0, 0.5, None),
    (["awful", "awfully"], -2.0, 0.5, None),
    (["horrible", "horribly"], -2.0, 0.5, None),
    (["appalling", "appallingly"], -2.1, 0.45, None),
    (["pathetic", "pathetically"], -1.9, 0.5, None),
    (["inept", "ineptly", "ineptitude"], -1.9, 0.5, None),
    (["incompetent", "incompetently", "incompetence"], -1.9, 0.5, None),
    (["amateurish", "amateurishly"], -1.8, 0.55, None),
    (["embarrassing", "embarrassingly", "embarrasses", "embarrassed", "embarrass"], -1.8, 0.55, None),
    (["humiliating", "humiliatingly"], -1.9, 0.5, None),
    (["shameful", "shamefully"], -1.8, 0.5, None),
    (["laughable", "laughably"], -1.8, 0.55, None),
    (["ludicrous", "ludicrously"], -1.9, 0.5, None),
    (["idiotic", "idiotically"], -2.0, 0.45, None),
    (["moronic", "moronically"], -2.0, 0.45, None),
    (["imbecilic"], -2.0, 0.45, None),
    (["disgraceful", "disgracefully"], -1.9, 0.5, None),
    (["deplorable", "deplorably"], -1.9, 0.5, None),
    (["woeful", "woefully"], -1.9, 0.5, None),
    (["lamentable", "lamentably"], -1.8, 0.5, None),
    (["worthless", "worthlessly"], -2.0, 0.45, None),
]

# ---------------------------------------------------------------------------
# VERY STRONG NEGATIVE
# ---------------------------------------------------------------------------
VERY_STRONG_NEGATIVE = [
    (["abysmal", "abysmally"], -2.6, 0.3, None),
    (["atrocious", "atrociously"], -2.6, 0.3, None),
    (["dreadful", "dreadfully"], -2.5, 0.3, None),
    (["disastrous", "disastrously"], -2.5, 0.3, None),
    (["catastrophic", "catastrophically"], -2.5, 0.3, None),
    (["unbearable", "unbearably"], -2.4, 0.35, None),
    (["excruciating", "excruciatingly"], -2.5, 0.3, None),
    (["torturous", "torturously"], -2.4, 0.35, None),
    (["agonizing", "agonizingly"], -2.4, 0.35, None),
    (["insufferable", "insufferably"], -2.3, 0.35, None),
    (["repugnant", "repugnantly"], -2.4, 0.3, None),
    (["repulsive", "repulsively"], -2.4, 0.3, None),
    (["revolting", "revoltingly"], -2.4, 0.3, None),
    (["vile", "vilely"], -2.4, 0.3, None),
    (["despicable", "despicably"], -2.3, 0.35, None),
    (["loathsome"], -2.4, 0.3, None),
    (["detestable"], -2.3, 0.3, None),
    (["nauseating", "nauseatingly"], -2.4, 0.3, None),
    (["horrendous", "horrendously"], -2.5, 0.3, None),
    (["egregious", "egregiously"], -2.3, 0.35, None),
]

# ---------------------------------------------------------------------------
# SUPERLATIVE NEGATIVE
# ---------------------------------------------------------------------------
SUPERLATIVE_NEGATIVE = [
    (["worst"], -3.0, 0.25, None),
    (["garbage"], -2.9, 0.25, None),
    (["trash", "trashy"], -2.7, 0.3, None),
    (["travesty"], -2.9, 0.25, None),
    (["abomination"], -3.0, 0.25, None),
    (["disaster"], -2.6, 0.3, "as a noun describing the whole film"),
    (["catastrophe"], -2.7, 0.3, None),
    (["atrocity"], -2.8, 0.25, None),
    (["trainwreck", "train-wreck"], -2.8, 0.25, None),
    (["dreck"], -2.8, 0.25, None),
    (["tripe"], -2.7, 0.25, None),
    (["drivel"], -2.7, 0.25, None),
    (["junk"], -2.3, 0.4, None),
    (["dud", "duds"], -2.2, 0.4, None),
    (["flop", "flops"], -2.0, 0.45, None),
    (["bomb", "bombs"], -2.0, 0.45, "box-office/critical-failure sense"),
]

# ---------------------------------------------------------------------------
# BATCH 2: frequency-gap top-up. Found by listing every vocab word NOT yet
# covered by the tiers above (or by structural_words.yaml), sorted by train
# frequency, and reading through it for sentiment-bearing words I had
# missed -- not by guessing forms and hoping they exist (batch 1's method,
# which had a 72% hit rate). This batch is sourced directly from the
# vocabulary, so it should match near 100%; expand_lexicon.py still
# verifies every form. Values follow the same tiers as above, by the same
# kind of judgment, just applied to words found this way round.
#
# A number of frequent words were deliberately left OUT despite looking
# sentiment-adjacent, because their polarity is genre- or context-flipped
# rather than quality-flipped, and a single word-level weight would be
# more likely to mislead than help: scary, frightening, terrifying,
# chilling-as-horror-praise (chilling kept, mild, see below), shocking,
# outrageous, dark, tension, guilty, hot, fat, sick, complicated. See
# RATIONALE.md for the general policy on genre-appropriate-affect words.
# ---------------------------------------------------------------------------
GAP_FILL_STRONG_POSITIVE = [
    (["splendid"], 2.0, 0.45, None),
    (["fantastic", "fantastically"], 1.9, 0.5, None),
    (["treasure", "treasures"], 1.8, 0.55, "noun: 'a hidden treasure'"),
    (["top-notch"], 1.8, 0.55, None),
    (["intoxicating", "intoxicatingly"], 1.6, 0.55, None),
    (["profound", "profoundly"], 1.5, 0.6, None),
    (["revelatory"], 1.5, 0.55, None),
    (["irresistible", "irresistibly"], 1.5, 0.6, None),
    (["magical", "magic"], 1.4, 0.6, None),
    (["heartbreaking", "heartbreakingly"], 1.4, 0.6, None),
    (["engrossing"], 1.5, 0.6, None),
    (["absorbing"], 1.4, 0.6, None),
    (["achievement", "achieves", "achieve"], 1.3, 0.65, None),
    (["delicious", "deliciously"], 1.3, 0.65, None),
    (["blast"], 1.3, 0.65, "noun, 'a blast' = great fun"),
    (["sensational", "sensationally"], 1.3, 0.65, None),
    (["aplomb"], 1.3, 0.6, None),
    (["feat"], 1.1, 0.65, "'quite a feat'"),
    (["wow"], 1.3, 0.6, None),
    (["thought-provoking"], 1.3, 0.6, None),
    (["first-rate"], 1.8, 0.55, None),
    (["charmer", "charmers"], 1.2, 0.65, None),
    (["richer"], 1.3, 0.65, "comparative of rich"),
    (["smarter"], 1.1, 0.65, "comparative of smart"),
]

GAP_FILL_SOLID_POSITIVE = [
    (["eloquent", "eloquently"], 1.2, 0.65, None),
    (["verve"], 1.2, 0.6, None),
    (["soulful", "soulfully"], 1.2, 0.65, None),
    (["glory"], 1.2, 0.6, "noun"),
    (["splendor"], 1.4, 0.6, None),
    (["evocative", "evocatively"], 1.1, 0.65, None),
    (["lyrical", "lyrically"], 1.1, 0.65, None),
    (["meaningful", "meaningfully"], 1.1, 0.7, None),
    (["involving"], 1.1, 0.65, None),
    (["grandeur"], 1.0, 0.65, None),
    (["insights"], 0.8, 0.7, "plural of insight"),
    (["happy", "happily", "happiness"], 1.0, 0.75, None),
    (["winner", "winners"], 1.0, 0.7, "praise sense: 'the film is a winner'"),
    (["admire", "admires", "admired", "admiring"], 1.0, 0.7, None),
    (["superior"], 1.3, 0.65, None),
    (["tender", "tenderly", "tenderness"], 0.9, 0.7, None),
    (["enthusiasm", "enthusiastic", "enthusiastically"], 1.0, 0.7, None),
    (["excitement", "exciting", "excitingly"], 1.0, 0.7, None),
    (["undeniable", "undeniably"], 1.0, 0.7, None),
    (["strength", "strengths"], 0.8, 0.7, None),
    (["compassion", "compassionate"], 1.0, 0.7, None),
    (["vital", "vitally"], 1.0, 0.65, None),
    (["potent", "potently"], 1.1, 0.65, None),
    (["cohesive"], 0.8, 0.7, None),
    (["capture", "captures", "captured", "capturing"], 0.9, 0.7, "praise verb: successfully depicts"),
    (["delivers", "delivered", "delivering", "deliver"], 1.0, 0.7, "'delivers the goods'"),
]

GAP_FILL_PLAIN_POSITIVE_2 = [
    (["positive", "positively"], 0.9, 0.75, None),
    (["intricate", "intricately"], 0.9, 0.7, None),
    (["hopeful"], 0.8, 0.75, "'hopefully' excluded: as a sentence-adverb it marks the speaker's stance, not the film's quality"),
    (["playful", "playfully"], 0.8, 0.75, None),
    (["sophisticated"], 1.0, 0.7, None),
    (["striking", "strikingly"], 1.0, 0.7, None),
    (["poetic", "poetically", "poetry"], 0.9, 0.7, None),
    (["resourceful"], 0.9, 0.7, None),
    (["uncompromising"], 0.9, 0.65, None),
    (["integrity"], 0.9, 0.65, None),
    (["good-natured"], 0.9, 0.75, None),
    (["proud", "proudly"], 0.9, 0.7, None),
    (["committed"], 0.8, 0.7, "praise for performance commitment"),
    (["inspiration", "inspirational"], 0.9, 0.7, None),
    (["celebration"], 0.7, 0.75, None),
    (["wise", "wisely"], 0.8, 0.7, None),
    (["flair"], 1.0, 0.65, None),
    (["detailed"], 0.6, 0.75, None),
    (["realistic", "realistically"], 0.8, 0.7, None),
    (["attractive", "attractively"], 0.8, 0.75, None),
    (["bright", "brightly"], 0.8, 0.75, None),
    (["worthy", "worthily"], 0.9, 0.7, None),
    (["clarity"], 0.6, 0.75, None),
    (["refreshing", "refreshingly"], 1.2, 0.65, None),
    (["inspiring"], 1.4, 0.6, None),
    (["treat"], 0.8, 0.75, "noun, 'a real treat'"),
    (["humorous", "humorously"], 0.9, 0.75, None),
    (["nicely"], 0.6, 0.75, None),
    (["crafted"], 0.9, 0.7, "root of well-crafted, standalone praise sense"),
    (["welcome"], 0.8, 0.75, "'a welcome surprise'"),
    (["timely"], 0.6, 0.75, None),
    (["accessible"], 0.7, 0.75, None),
    (["appreciate", "appreciates", "appreciated"], 0.8, 0.75, None),
    (["coherent", "coherently"], 0.7, 0.75, None),
    (["faithful", "faithfully"], 0.6, 0.75, "praise sense for an adaptation"),
    (["suspenseful"], 1.0, 0.7, None),
]

GAP_FILL_MILD_POSITIVE_2 = [
    (["nice", "nicer"], 0.7, 0.8, None),
    (["chilling", "chillingly"], 0.5, 0.75, "critic-praise sense: effectively unsettling, not a flaw"),
    (["biting", "bitingly"], 0.7, 0.75, "'biting wit'"),
    (["wry", "wryly"], 0.6, 0.75, None),
    (["merit", "merits"], 0.6, 0.75, None),
    (["presence"], 0.5, 0.75, "'has presence'"),
    (["ambition", "ambitions"], 0.6, 0.75, None),
    (["sensual", "sensually"], 0.7, 0.75, None),
    (["mature", "maturely"], 0.6, 0.75, "craft-praise sense"),
    (["tasty"], 0.6, 0.8, None),
    (["offbeat"], 0.6, 0.75, None),
    (["homage"], 0.4, 0.8, None),
    (["challenging"], 0.5, 0.75, "intellectually-engaging sense"),
    (["substance"], 0.6, 0.75, "'has substance'"),
    (["depth"], 0.6, 0.75, "'has depth'"),
    (["empathy", "empathetic"], 0.6, 0.75, None),
    (["spare", "sparely"], 0.3, 0.8, "minimalist-praise sense"),
    (["spooky"], 0.3, 0.8, "genre-appropriate-fun sense"),
    (["unforced"], 0.8, 0.7, None),
    (["unusual"], 0.3, 0.8, None),
    (["quick", "quickness"], 0.3, 0.8, "pacing/wit sense"),
    (["comfortable", "comfortably"], 0.3, 0.8, None),
    (["mysterious"], 0.4, 0.8, None),
    (["exotic"], 0.4, 0.8, None),
    (["straightforward"], 0.2, 0.8, None),
    (["potential"], 0.3, 0.8, "promise sense"),
    (["hip"], 0.3, 0.8, None),
    (["demanding"], 0.3, 0.8, "intellectually-ambitious sense"),
    (["adventurous"], 0.8, 0.75, None),
    (["capable", "capably"], 0.7, 0.75, None),
    (["fanciful"], 0.5, 0.8, None),
    (["amused", "amuses"], 0.7, 0.75, None),
    (["artistic", "artistically"], 0.6, 0.75, None),
    (["credible", "credibly"], 0.8, 0.75, None),
    (["startling", "startlingly"], 0.8, 0.7, None),
    (["overwhelming", "overwhelmingly"], 0.6, 0.7, "dominant collocation is 'overwhelmingly positive'"),
    (["sweeping"], 0.7, 0.75, "'a sweeping epic'"),
]

GAP_FILL_MILD_NEGATIVE_2 = [
    (["tough", "toughness"], -0.3, 0.8, None),
    (["weird", "weirdly", "weirdness"], -0.3, 0.8, None),
    (["strange", "strangely"], -0.2, 0.8, None),
    (["odd", "oddly"], -0.3, 0.8, None),
    (["crazy"], -0.3, 0.8, None),
    (["plain"], -0.4, 0.8, "'plain and boring'"),
    (["bizarre", "bizarrely"], -0.5, 0.8, None),
    (["mainstream"], -0.3, 0.8, "dismissive-contrast-with-indie sense"),
    (["standard"], -0.5, 0.8, "'standard fare'"),
    (["basic"], -0.3, 0.8, None),
    (["chaotic", "chaotically"], -0.5, 0.75, None),
    (["angst"], -0.3, 0.8, None),
    (["b-movie"], -0.4, 0.8, None),
    (["difficult"], -0.4, 0.8, "'a difficult watch'"),
    (["limited"], -0.4, 0.8, None),
    (["convenient", "conveniently"], -0.4, 0.75, "'a convenient plot device'"),
    (["mixed"], -0.4, 0.8, "'mixed results'"),
    (["loud", "loudly"], -0.3, 0.8, None),
    (["flashy", "flashily"], -0.5, 0.8, None),
    (["heavy"], -0.5, 0.8, "tone/pacing complaint"),
    (["determined", "determination"], 0.4, 0.8, "positive: a determined character/performance, not a complaint"),
    (["sentimental"], -0.3, 0.8, None),
    (["typical"], -0.4, 0.8, "'typical Hollywood fare'"),
    (["perverse", "perversely"], -0.6, 0.75, None),
    (["cynical", "cynically"], -0.6, 0.75, None),
    (["despair"], -0.5, 0.75, None),
    (["exploitation"], -0.5, 0.75, "'cheap exploitation'"),
    (["dirty"], -0.3, 0.8, None),
    (["ordinary"], -0.6, 0.8, None),
    (["conventional", "conventionally"], -0.6, 0.75, None),
    (["unfortunately"], -0.6, 0.8, "discourse marker flagging a negative assessment follows"),
    (["sadly"], -0.5, 0.8, "discourse marker flagging a negative assessment follows"),
    (["forgotten"], -0.6, 0.75, None),
    (["bitter", "bitterly"], -0.6, 0.75, None),
    (["missed"], -0.6, 0.75, "'a missed opportunity'"),
    (["hastily"], -0.6, 0.75, None),
    (["wrong"], -0.8, 0.8, None),
    (["desperate", "desperation"], -0.7, 0.75, None),
    (["pretends", "pretended", "pretend", "pretending"], -0.7, 0.75, "dismissive: 'pretends to be deep'"),
    (["unfaithful"], -0.8, 0.75, "'an unfaithful adaptation'"),
    (["trouble", "troubled", "troubling"], -0.7, 0.75, None),
    (["inevitable", "inevitably"], -0.3, 0.8, "predictability complaint"),
    (["unlikely"], -0.2, 0.8, None),
    (["chilly"], -0.3, 0.8, None),
]

GAP_FILL_PLAIN_NEGATIVE_2 = [
    (["manipulative", "manipulatively"], -0.8, 0.75, None),
    (["painfully"], -0.8, 0.7, None),
    (["murky", "murkily"], -0.8, 0.75, None),
    (["false", "falsely"], -0.9, 0.75, None),
    (["excuse", "excuses"], -0.8, 0.75, "'just an excuse to...'"),
    (["sketchy", "sketchily"], -0.8, 0.75, None),
    (["phony", "phoniness"], -1.2, 0.65, None),
    (["farce"], -0.8, 0.75, "dismissive sense, not the neutral genre label"),
    (["shame"], -1.0, 0.7, "'a shame that...'"),
    (["self-conscious", "self-consciously"], -0.9, 0.7, None),
    (["broken"], -0.7, 0.75, None),
    (["embarrassment", "embarrassments"], -1.3, 0.65, None),
    (["sour", "sourly"], -0.9, 0.75, None),
    (["unpleasant", "unpleasantly"], -1.2, 0.7, None),
    (["suffers", "suffered"], -1.1, 0.7, "'suffers from a weak script'"),
    (["unnecessary", "unnecessarily"], -0.9, 0.75, None),
    (["artificial", "artificially"], -1.0, 0.7, None),
    (["tortured"], -1.0, 0.7, None),
    (["brutal", "brutally"], -0.9, 0.7, "default negative; see RATIONALE.md for the praise-sense exception"),
    (["recycled"], -1.0, 0.7, None),
    (["fake", "fakery"], -1.1, 0.7, None),
    (["nasty", "nastily"], -1.0, 0.7, None),
    (["mess"], -1.3, 0.65, "noun: 'the film is a mess'"),
    (["lame", "lamely"], -1.3, 0.7, None),
    (["ugly", "ugliness"], -1.1, 0.7, None),
    (["routine"], -0.9, 0.75, "'a routine thriller'"),
    (["unfocused"], -1.1, 0.7, None),
    (["nightmare"], -1.3, 0.65, "'a nightmare to sit through'"),
    (["smug", "smugness", "smugly"], -1.2, 0.65, None),
    (["crude", "crudely", "crudeness"], -1.1, 0.7, None),
    (["melodramatic", "melodrama"], -1.0, 0.7, None),
    (["choppy", "choppily"], -1.1, 0.7, None),
    (["juvenile"], -1.1, 0.65, None),
    (["soulless"], -1.5, 0.6, None),
    (["direct-to-video"], -1.2, 0.65, None),
    (["doze"], -1.2, 0.65, "'doze off'"),
    (["empty", "emptiness"], -1.1, 0.7, None),
    (["frustration", "frustrated", "frustrates"], -1.0, 0.7, None),
    (["frustrating", "frustratingly"], -1.2, 0.7, None),
    (["simplistic", "simplistically"], -1.1, 0.7, None),
    (["shapeless"], -1.1, 0.65, None),
    (["paper-thin"], -1.4, 0.6, None),
    (["vulgar", "vulgarity"], -1.3, 0.65, None),
    (["crass", "crassly"], -1.3, 0.65, None),
    (["distasteful"], -1.3, 0.6, None),
    (["cloying", "cloyingly"], -1.2, 0.65, None),
    (["banal", "banality"], -1.3, 0.65, None),
    (["dumbed-down"], -1.3, 0.65, None),
    (["obnoxious", "obnoxiously"], -1.5, 0.6, None),
    (["witless"], -1.5, 0.6, None),
    (["failure", "failures"], -1.3, 0.7, None),
    (["offensive", "offensively"], -1.3, 0.7, None),
    (["endless"], -1.0, 0.75, None),
    (["retread"], -1.3, 0.65, None),
    (["pointless", "pointlessly"], -1.3, 0.7, None),
    (["rip-off"], -1.6, 0.6, None),
    (["lousy"], -1.8, 0.55, None),
    (["suck", "sucks", "sucked", "sucking"], -1.8, 0.55, None),
    (["crap", "crappy"], -2.3, 0.4, None),
    (["skip"], -1.3, 0.65, "'skip this one'"),
    (["mindless", "mindlessly"], -1.1, 0.7, None),
    (["poorly"], -1.0, 0.75, None),
    (["flaws", "flaw"], -0.9, 0.75, None),
]

GAP_FILL_MISC_SMALL = [
    (["manages", "managed", "manage"], 0.4, 0.75, "'manages to be funny despite...' -- modest-achievement sense"),
    (["touched"], 1.1, 0.65, "joins the touching/moving emotional-sense family"),
    (["rises", "rise"], 1.3, 0.6, "'rises above its flaws' -- joins the soars family"),
    (["loved", "loves", "love", "loving"], 1.3, 0.7, None),
    (["liked", "likes"], 0.7, 0.75, "bare 'like' excluded: too overloaded with the similarity preposition ('like a conspiracy'); the inflected forms here are unambiguously the verb"),
]

# ---------------------------------------------------------------------------
# FAMILIES combines every tier above into one list the expander consumes.
# ---------------------------------------------------------------------------
FAMILIES = (
    SUPERLATIVE_POSITIVE
    + VERY_STRONG_POSITIVE
    + STRONG_POSITIVE
    + SOLID_POSITIVE
    + PLAIN_POSITIVE
    + MILD_POSITIVE
    + MILD_NEGATIVE
    + PLAIN_NEGATIVE
    + SOLID_NEGATIVE
    + STRONG_NEGATIVE
    + VERY_STRONG_NEGATIVE
    + SUPERLATIVE_NEGATIVE
    + GAP_FILL_STRONG_POSITIVE
    + GAP_FILL_SOLID_POSITIVE
    + GAP_FILL_PLAIN_POSITIVE_2
    + GAP_FILL_MILD_POSITIVE_2
    + GAP_FILL_MILD_NEGATIVE_2
    + GAP_FILL_PLAIN_NEGATIVE_2
    + GAP_FILL_MISC_SMALL
)

