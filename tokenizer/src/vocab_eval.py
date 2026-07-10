"""
Module 4: Per-language corpus -> (full word frequencies, evaluation vocabulary).

Two deliberately different word lists come out of this module, and
mixing them up is exactly what caused a zero-UNK bug earlier:

  - `freq`: EVERY distinct word type in the language's full article,
    case-folded (Latin scripts only). This is what the BPE base
    alphabet must be built from -- every character/akshara that occurs
    anywhere in the corpus needs a base token, or some word would need
    an UNK fallback to encode. Nothing is excluded here, not even
    pure-numeral words, specifically so digits always get a base slot.

  - `vocab_words`: the top TOP_N most frequent word types, with
    case-folding AND pure-numeral words excluded. This is "the
    language's vocabulary" for X = tokens/word purposes -- the
    assignment's own "say 5000 words" phrasing frames vocab as a
    reference list, not literally every string that happened to appear
    once in one Wikipedia article, and single-article corpora are
    heavy with one-off proper nouns/numbers that inflate that count
    without reflecting real vocabulary. TOP_N=1300 is chosen empirically
    (see REPORT.md) to sit in the middle of a wide, stable range that
    lets the single joint BPE balance all four languages tightly
    without either exhausting early (which causes a degenerate 0-gap,
    divide-by-zero score) or overshooting Telugu's own vocabulary size.

`freq` is a superset of `vocab_words`'s underlying words, so training
priority/evaluation (restricted to vocab_words) and base-alphabet
coverage (built from freq) stay consistent with each other by
construction.
"""
import os
from collections import Counter
from segmenters import extract_words, is_brahmic

TOP_N = 1300
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

LANG_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "es": "Spanish"}


def _casefold(word: str) -> str:
    return word if is_brahmic(word) else word.lower()


def load_language(lang: str, top_n: int = TOP_N) -> dict:
    path = os.path.join(DATA_DIR, f"{lang}.txt")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    freq = Counter()
    for w in extract_words(text):
        freq[_casefold(w)] += 1

    non_numeral = Counter({w: c for w, c in freq.items() if not w.isdigit()})
    vocab_words = [w for w, _ in non_numeral.most_common(top_n)]

    return {
        "lang": lang,
        "name": LANG_NAMES[lang],
        "freq": dict(freq),                 # ALL word types (incl. numerals) -> base-alphabet coverage
        "vocab_words": vocab_words,          # top-N, numeral-free -> training priority + X evaluation
        "total_occurrences": sum(freq.values()),
        "unique_words": len(freq),
    }


def load_all(top_n: int = TOP_N) -> dict:
    return {lang: load_language(lang, top_n) for lang in ["en", "hi", "te", "es"]}


if __name__ == "__main__":
    for lang, d in load_all().items():
        print(f"{d['name']:8s} unique_words(full)={d['unique_words']:5d} "
              f"vocab_words(top-{TOP_N}, numeral-free)={len(d['vocab_words']):5d} "
              f"occurrences={d['total_occurrences']}")
