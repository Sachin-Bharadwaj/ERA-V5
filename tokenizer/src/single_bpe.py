"""
Module 5 (replaces allocator.py): a genuinely SINGLE joint BPE.

One BPETokenizer instance. One combined training state (every word from
every language lives in the same dict, keyed by its own string -- scripts
are disjoint so this never collides except for legitimately shared tokens
like ASCII digits/abbreviations embedded in Hindi/Telugu prose, which
*should* be shared). One growing vocab/merge list. One pair-frequency
counter, run to completion for a single 10,000-token budget.

The only per-language notion that exists is the *priority rule* used to
pick which pair the single counter should merge next -- exactly the kind
of engineered training-data weighting real multilingual tokenizers use
(mBERT/XLM-R exponential language-sampling) to keep one script from
starving another. Concretely, two phases of the same loop:

  Phase 1 (hard constraint): restrict pair-frequency counting to English's
  own words only, and keep merging English's best pair, until English's
  tokens/word ratio drops below 1.2.

  Phase 2 (fairness objective): for whatever budget remains, restrict
  pair-frequency counting to whichever language currently has the worst
  (highest) ratio -- across all four languages, English included -- and
  merge its best pair. This directly minimizes max(X) - min(X), which is
  what the assignment's score (1000 / (X4 - X1)) rewards.

Regardless of which phase picked a pair, the merge itself is *applied to
the entire combined state* (every language), not just the phase's
priority subset -- so a Spanish word sharing a substring with an
English-motivated merge benefits for free, and the training state never
drifts out of sync with what tok.encode() would do on held-out text.

Base-alphabet coverage vs. training/eval priority are kept deliberately
separate: `state` (and therefore `tok.vocab`'s base symbols) is
initialized from `lang_data[lang]["freq"]` -- EVERY word in the full
corpus -- so every character/akshara that occurs anywhere always has a
base token. `words_by_lang` (which pairs get priority in phase 1/2, and
which words the final X ratios are computed over) uses the smaller,
curated `lang_data[lang]["vocab_words"]` instead. Conflating the two
(restricting the base alphabet itself to only the curated list) is what
would produce an actual UNK: a rare word excluded from vocab_words could
then contain a character with no base token at all. See
`verify_no_unk` below for the check that this never happens.
"""
from collections import Counter
from bpe import BPETokenizer
from segmenters import segment_word, tokenize_full_text, is_word_pretoken, PUNCT_WHITESPACE_SEED


def ratio(state: dict, words) -> float:
    if not words:
        return float("inf")
    return sum(len(state[w]) for w in words) / len(words)


def _best_pair(state: dict, freq: dict, words) -> tuple:
    pairs = Counter()
    for w in words:
        symbols = state[w]
        wf = freq[w]
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += wf
    if not pairs:
        return None
    return max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _apply_merge_globally(tok: BPETokenizer, state: dict, pair: tuple) -> bool:
    """Apply `pair` to every word in the combined state. Returns True if
    anything actually changed (i.e. this was a genuinely new merge)."""
    a, b = pair
    changed = False
    for w, symbols in state.items():
        if a in symbols and b in symbols:
            new_symbols = BPETokenizer._merge_symbol(symbols, pair)
            if new_symbols != symbols:
                state[w] = new_symbols
                changed = True
    if changed:
        merged = a + b
        tok.merge_rank[pair] = len(tok.merges)
        tok.merges.append(pair)
        tok.vocab.add(merged)
    return changed


def train_single_bpe(lang_data: dict, total_budget: int = 10000, en_target: float = 1.2):
    tok = BPETokenizer("multilingual-single-bpe")

    # Seeded before anything else, so it's costed against the budget from
    # the start: ASCII punctuation/Markdown symbols, whitespace, and a
    # couple of script-specific marks. Without these, the tokenizer only
    # ever knows about word characters -- it has no token at all for an
    # apostrophe, a comma, a period, or a space, so it cannot encode (let
    # alone decode) real running text, only isolated word strings. See
    # `bpe.py::BPETokenizer.encode_text` for the full-text path this
    # unlocks, and `segmenters.py::PUNCT_WHITESPACE_SEED` for the set.
    tok.vocab.update(PUNCT_WHITESPACE_SEED)

    combined_freq = Counter()
    for d in lang_data.values():
        combined_freq.update(d["freq"])   # sums counts for any string shared across languages
    combined_freq = dict(combined_freq)

    state = tok.init_vocab(combined_freq, segment_word)   # full-corpus coverage -> no UNK possible
    words_by_lang = {lang: list(d["vocab_words"]) for lang, d in lang_data.items()}

    remaining = total_budget - len(tok.vocab)
    if remaining < 0:
        raise ValueError(f"Base alphabets alone need {len(tok.vocab)} slots, over budget {total_budget}.")

    history = []

    # ---- Phase 1: hard constraint -- force English below en_target -----
    en_words = words_by_lang["en"]
    while remaining > 0 and ratio(state, en_words) >= en_target:
        pair = _best_pair(state, combined_freq, en_words)
        if pair is None:
            break
        if _apply_merge_globally(tok, state, pair):
            remaining -= 1
        history.append({"phase": 1, "driver": "en", "pair": pair,
                         "en_ratio": ratio(state, en_words), "slots_used": total_budget - remaining})

    phase1_merges = len(tok.merges)

    # ---- Phase 2: minimize max(X) - min(X) over all four with what's left
    exhausted = set()
    while remaining > 0 and len(exhausted) < len(lang_data):
        current = {l: ratio(state, words_by_lang[l]) for l in lang_data if l not in exhausted}
        worst = max(current, key=current.get)
        pair = _best_pair(state, combined_freq, words_by_lang[worst])
        if pair is None:
            exhausted.add(worst)
            continue
        if _apply_merge_globally(tok, state, pair):
            remaining -= 1
        history.append({"phase": 2, "driver": worst, "pair": pair,
                         "ratio_after": ratio(state, words_by_lang[worst]), "slots_used": total_budget - remaining})

    return {
        "tokenizer": tok,
        "state": state,
        "words_by_lang": words_by_lang,
        "history": history,
        "phase1_merges": phase1_merges,
        "phase2_merges": len(tok.merges) - phase1_merges,
        "slots_used": total_budget - remaining,
        "budget": total_budget,
    }


def encode_text(tok: BPETokenizer, text: str) -> list:
    """Text-level encode: word runs through segment_word + learned merges,
    every other character passed through literally. See
    `bpe.py::BPETokenizer.encode_text` for why this always round-trips."""
    return tok.encode_text(text, segment_word, tokenize_full_text, is_word_pretoken)


def verify_roundtrip(tok: BPETokenizer, samples: list) -> dict:
    """
    For each sample string, encode_text then decode, and check that every
    VISIBLE NON-WHITESPACE character survives in order (whitespace may
    legitimately be reshaped -- e.g. collapsed runs -- but nothing here
    actually does that; decode is exact concatenation, so in practice
    whitespace round-trips exactly too). This is the concrete check behind
    the "faithful roundtrip" gate: decode(encode_text(text)) must preserve
    the same visible non-whitespace characters as `text`.
    """
    results = []
    all_ok = True
    for text in samples:
        tokens = encode_text(tok, text)
        decoded = BPETokenizer.decode(tokens)
        visible_in = "".join(ch for ch in text if not ch.isspace())
        visible_out = "".join(ch for ch in decoded if not ch.isspace())
        ok = visible_in == visible_out
        exact = decoded == text
        all_ok = all_ok and ok
        results.append({
            "text": text, "decoded": decoded, "n_tokens": len(tokens),
            "visible_chars_preserved": ok, "exact_roundtrip": exact,
        })
    return {"samples": results, "all_ok": all_ok}


def verify_no_unk(tok: BPETokenizer, lang_data: dict) -> dict:
    """
    Re-segment EVERY word in every language's full corpus (not just the
    top-N vocab_words used for training) and confirm every atomic unit
    it decomposes into is present in the tokenizer's vocab. This is the
    concrete, checkable claim behind "no UNK token in any of the four
    languages": since `tok.vocab`'s base symbols were built from the same
    `freq` dicts checked here, this should always pass by construction --
    running it is what turns that guarantee from an argument into a fact.
    """
    report = {}
    all_ok = True
    for lang, d in lang_data.items():
        missing = set()
        offending_words = []
        for w in d["freq"]:
            units = segment_word(w)
            bad = [u for u in units if u not in tok.vocab]
            if bad:
                missing.update(bad)
                offending_words.append((w, bad))
        ok = len(missing) == 0
        all_ok = all_ok and ok
        report[lang] = {
            "words_checked": len(d["freq"]),
            "ok": ok,
            "missing_units": sorted(missing),
            "offending_words": offending_words[:10],
        }
    report["all_ok"] = all_ok
    return report
