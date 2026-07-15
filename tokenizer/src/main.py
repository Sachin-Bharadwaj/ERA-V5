"""
Module 6: End-to-end pipeline orchestrator.

corpora -> vocab_eval (full-coverage freq + top-N curated vocab_words)
-> single_bpe (ONE joint 10,000-token BPE, seeded with a punctuation/
whitespace base alphabet so real prose -- not just isolated words --
can be encoded and decoded; English forced below 1.2, remaining budget
spent minimizing max(X)-min(X) across all four languages) -> zero-UNK
verification -> faithful-roundtrip verification -> final X1..X4 + score
-> JSON/vocab/merges dumps + REPORT.md.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from vocab_eval import load_all, TOP_N
from single_bpe import train_single_bpe, ratio, verify_no_unk, verify_roundtrip
from segmenters import segment_word

ROUNDTRIP_SAMPLES = [
    "India's population is 1,428,627,663.",
    "# Heading\n\nSome *emphasis* and a [link](url), plus `code`.",
    "भारत की जनसंख्या 1,428,627,663 है।",
    "¿Cuál es la población de la India? ¡Más de mil millones!",
    "తెలుగు ప్రజల సంఖ్య 9,00,00,000 కి పైగా ఉంది.",
]

TOTAL_BUDGET = 10000
EN_TARGET = 1.2
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def pick_examples(state, freq, words, n=6):
    ranked = sorted(words, key=lambda w: -freq[w])
    out = []
    seen_lengths = set()
    for w in ranked:
        toks = state[w]
        if len(toks) not in seen_lengths or len(out) < n:
            out.append({"word": w, "tokens": toks, "n_tokens": len(toks), "corpus_freq": freq[w]})
            seen_lengths.add(len(toks))
        if len(out) >= n:
            break
    return out


def dump_vocab_and_merges(tok, out_dir):
    # GPT-2-style vocab.json: token string -> integer id, base symbols first
    # (in an arbitrary but stable order), then merges in the order learned.
    base_symbols = sorted(tok.vocab - {a + b for a, b in tok.merges})
    ordered_tokens = list(base_symbols) + [a + b for a, b in tok.merges]
    vocab_ids = {tok_str: i for i, tok_str in enumerate(ordered_tokens)}
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_ids, f, ensure_ascii=False, indent=1)

    merges_list = [{"rank": i, "left": a, "right": b, "result": a + b}
                    for i, (a, b) in enumerate(tok.merges)]
    with open(os.path.join(out_dir, "merges.json"), "w", encoding="utf-8") as f:
        json.dump(merges_list, f, ensure_ascii=False, indent=1)

    return len(ordered_tokens), len(merges_list)


def main():
    t0 = time.time()
    print(f"Loading corpora + extracting vocab (top-{TOP_N} curated vocab_words per language)...")
    lang_data = load_all()
    for lang, d in lang_data.items():
        print(f"  {d['name']:8s} unique_words(full)={d['unique_words']:5d}  "
              f"vocab_words={len(d['vocab_words']):5d}")

    print(f"\nTraining ONE joint BPE, budget={TOTAL_BUDGET}, "
          f"English forced below {EN_TARGET}, remainder balances the rest...")
    result = train_single_bpe(lang_data, total_budget=TOTAL_BUDGET, en_target=EN_TARGET)
    tok = result["tokenizer"]
    state = result["state"]
    print(f"  slots used: {result['slots_used']} / {result['budget']}")
    print(f"  phase 1 merges (forcing English < {EN_TARGET}): {result['phase1_merges']}")
    print(f"  phase 2 merges (balancing all four with the rest): {result['phase2_merges']}")

    print("\nVerifying zero UNK across every word in all four FULL corpora...")
    unk_report = verify_no_unk(tok, lang_data)
    for lang in ["en", "hi", "te", "es"]:
        r = unk_report[lang]
        print(f"  {lang}: checked {r['words_checked']:5d} words -> "
              f"{'OK, zero UNK' if r['ok'] else 'FAILED, missing units: ' + str(r['missing_units'])}")
    print(f"  ALL LANGUAGES ZERO-UNK: {unk_report['all_ok']}")

    print("\nVerifying faithful roundtrip (decode(encode_text(text)) == text)...")
    roundtrip_report = verify_roundtrip(tok, ROUNDTRIP_SAMPLES)
    for s in roundtrip_report["samples"]:
        status = "OK" if s["exact_roundtrip"] else "FAILED"
        print(f"  {status}: {s['text']!r} -> {s['n_tokens']} tokens")
        if not s["exact_roundtrip"]:
            print(f"    decoded: {s['decoded']!r}")
    print(f"  ALL ROUNDTRIP OK: {roundtrip_report['all_ok']}")

    summary = {}
    for lang, d in lang_data.items():
        words = result["words_by_lang"][lang]
        r = ratio(state, words)
        n_tokens = sum(len(state[w]) for w in words)
        base_units = set()
        for w in words:
            base_units.update(segment_word(w))
        summary[lang] = {
            "name": d["name"],
            "unique_words": len(words),
            "total_tokens": n_tokens,
            "ratio": r,
            "base_units": len(base_units),
            "examples": pick_examples(state, d["freq"], words),
        }

    order = sorted(summary.keys(), key=lambda l: summary[l]["ratio"])
    x_sorted = [(lang, summary[lang]["ratio"]) for lang in order]
    x_min = x_sorted[0][1]
    x_max = x_sorted[-1][1]
    score = 1000.0 / (x_max - x_min) if x_max > x_min else float("inf")

    print("\nFinal ratios (tokens / curated vocab word), sorted ascending:")
    for lang, r in x_sorted:
        print(f"  {summary[lang]['name']:8s} X={r:.4f}")
    print(f"\nEnglish ratio < {EN_TARGET} ? {summary['en']['ratio'] < EN_TARGET}")
    print(f"X_max - X_min = {x_max - x_min:.5f}")
    print(f"SCORE = 1000 / (X_max - X_min) = {score:.2f}")
    print(f"\nSingle joint tokenizer vocab size: {len(tok.vocab)} (== {TOTAL_BUDGET})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_vocab, n_merges = dump_vocab_and_merges(tok, OUTPUT_DIR)

    report = {
        "total_budget": TOTAL_BUDGET,
        "en_target": EN_TARGET,
        "top_n": TOP_N,
        "slots_used": result["slots_used"],
        "phase1_merges": result["phase1_merges"],
        "phase2_merges": result["phase2_merges"],
        "vocab_json_size": n_vocab,
        "merges_json_size": n_merges,
        "unk_check": {lang: {"words_checked": unk_report[lang]["words_checked"],
                              "ok": unk_report[lang]["ok"]} for lang in ["en", "hi", "te", "es"]},
        "unk_check_all_ok": unk_report["all_ok"],
        "roundtrip_check": roundtrip_report,
        "summary": summary,
        "x_sorted": x_sorted,
        "x_min": x_min,
        "x_max": x_max,
        "score": score,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nElapsed: {time.time() - t0:.1f}s. "
          f"Wrote output/results.json, output/vocab.json ({n_vocab} tokens), "
          f"output/merges.json ({n_merges} merges)")
    return report


if __name__ == "__main__":
    main()
