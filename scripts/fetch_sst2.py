"""Fetch ONLY the unlabeled sentence text of the SST-2 train split.

Integrity contract (see PROTOCOL.md and the project brief):
  - Downloads the official HF `stanfordnlp/sst2` TRAIN parquet file only.
    Validation and test are never requested.
  - Reads it with `pandas.read_parquet(path, columns=["sentence"])` so the
    `label` column is never decoded into memory or into any derived file.
  - Deletes the downloaded parquet immediately after the column-projected
    read, so the label column's bytes do not linger on disk either.
  - Writes only `data/train_sentences.txt`, one sentence per line, with a
    row count. No label value, count, or statistic is ever printed, logged,
    or written anywhere by this script.

Run: python scripts/fetch_sst2.py
"""

import os
import sys
import urllib.request

import pandas as pd

TRAIN_URL = (
    "https://huggingface.co/datasets/stanfordnlp/sst2/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PARQUET = os.path.join(DATA_DIR, "_raw_train.parquet")
OUT_SENTENCES = os.path.join(DATA_DIR, "train_sentences.txt")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Downloading TRAIN split only from: {TRAIN_URL}")
    urllib.request.urlretrieve(TRAIN_URL, RAW_PARQUET)

    # Column projection: pyarrow decodes ONLY the "sentence" column's pages.
    # The "label" column is never decoded, never materialized, never seen.
    df = pd.read_parquet(RAW_PARQUET, columns=["sentence"])
    assert list(df.columns) == ["sentence"], (
        "refusing to continue: expected exactly one column 'sentence', "
        f"got {list(df.columns)}"
    )

    sentences = df["sentence"].tolist()
    del df

    with open(OUT_SENTENCES, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s.replace("\n", " ").replace("\r", " ").strip() + "\n")

    # Remove the raw file immediately so the label column's bytes do not
    # persist on disk any longer than the single column-projected read above.
    os.remove(RAW_PARQUET)

    print(f"Wrote {len(sentences)} unlabeled sentences to {OUT_SENTENCES}")
    print("Raw parquet deleted. Label column was never read.")


if __name__ == "__main__":
    sys.exit(main())
