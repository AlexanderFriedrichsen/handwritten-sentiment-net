"""Build a label-free frequency vocabulary from the unlabeled SST-2 train text.

Reads ONLY data/train_sentences.txt (plain sentence text, no labels ever
touched it -- see fetch_sst2.py). Counts token frequency, keeps every token
with train frequency >= MIN_FREQ, and writes vocab/vocab.json: a committed,
reproducible artifact (word -> {index, train_frequency}) derived purely from
unlabeled text statistics. Index 0 is reserved for <UNK>.

Run: python scripts/build_vocab.py
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from handnet.tokenize import tokenize  # noqa: E402

MIN_FREQ = 2

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SENTENCES_PATH = os.path.join(DATA_DIR, "train_sentences.txt")
VOCAB_DIR = os.path.join(os.path.dirname(__file__), "..", "vocab")
VOCAB_PATH = os.path.join(VOCAB_DIR, "vocab.json")


def main() -> None:
    freq: Counter[str] = Counter()
    n_sentences = 0
    with open(SENTENCES_PATH, encoding="utf-8") as f:
        for line in f:
            freq.update(tokenize(line))
            n_sentences += 1

    kept = sorted(
        (w for w, c in freq.items() if c >= MIN_FREQ),
        key=lambda w: (-freq[w], w),
    )

    os.makedirs(VOCAB_DIR, exist_ok=True)
    vocab = {
        "source": "stanfordnlp/sst2 train split, sentence column only",
        "n_train_sentences": n_sentences,
        "min_freq": MIN_FREQ,
        "unk_index": 0,
        "size": len(kept) + 1,
        "words": {
            w: {"index": i + 1, "train_frequency": freq[w]}
            for i, w in enumerate(kept)
        },
    }
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=1, sort_keys=False)

    print(f"{n_sentences} sentences, {len(freq)} unique types, "
          f"{len(kept)} kept at min_freq>={MIN_FREQ} (+1 UNK) "
          f"-> {VOCAB_PATH}")


if __name__ == "__main__":
    sys.exit(main())
