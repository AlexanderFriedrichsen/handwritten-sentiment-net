"""Shared tokenizer used by both vocabulary construction and the encoder.

SST-2 (as distributed on HF `stanfordnlp/sst2`) is already PTB-tokenized:
lowercase, punctuation split into its own tokens (",", ".", "n't", "'s",
backtick/quote marks, "--"), hyphenated compounds kept whole. A plain
whitespace split is therefore sufficient and exact; no regex tokenizer is
needed. This was observed from the raw sentence text only (never labels).
"""

from __future__ import annotations


def tokenize(sentence: str) -> list[str]:
    return sentence.strip().lower().split()
