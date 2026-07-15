"""
Module 2: Script-aware pre-tokenization.

Two jobs, kept deliberately separate:

1. word extraction  -- cut raw prose into "word" strings without ever
   slicing through a combining mark or a ZWJ/ZWNJ joiner.
2. atomic-unit segmentation -- cut a word into the smallest units that
   BPE is allowed to operate on:
     - Latin scripts (en/es):  one Unicode codepoint == one grapheme,
       so the atomic unit is just the character (after NFC normalization
       composed marks and base letters are already fused, e.g. NFC turns
       "e" + COMBINING ACUTE into a single "é" codepoint).
     - Brahmic scripts (hi/te): the atomic unit is the *akshara*
       (orthographic syllable / extended grapheme cluster): an optional
       independent letter or a consonant, plus any nukta, any
       virama-joined consonant chain (conjuncts), any dependent vowel
       signs (matras) and any trailing modifier signs (candrabindu,
       anusvara, visarga). This is the unit a native reader perceives
       as "one character" and the unit screen readers/cursors move over.

Why one function serves both Devanagari and Telugu: Unicode allocates
Brahmic script blocks with deliberate positional parallelism -- the
same *relative offset* inside each script's 128-codepoint block carries
the same grammatical role. Concretely, in both blocks:
    offset 0x4D -> SIGN VIRAMA          (0x094D, 0x0C4D)
    offset 0x02 -> SIGN ANUSVARA        (0x0902, 0x0C02)
    offset 0x03 -> SIGN VISARGA         (0x0903, 0x0C03)
    offset 0x3C -> SIGN NUKTA           (0x093C, 0x0C3C)
    offset 0x3E.. -> dependent vowel signs (matras)
    0x05..0x14  -> independent vowels
    0x15..0x39  -> consonants
This lets a single offset/category-driven classifier work for every
Brahmic script without a per-script lookup table -- we only need to
special-case the two VIRAMA codepoints; everything else falls out of
`unicodedata.category()` (Mn/Mc = combining marks, Lo = base letters).
"""
import unicodedata
import regex  # PyPI `regex` module: supports \p{...} Unicode property syntax

# --- Unicode building blocks ------------------------------------------------

ZWNJ = "‌"
ZWJ = "‍"
JOINERS = {ZWNJ, ZWJ}

# The only script-specific hardcoding needed: VIRAMA sits at relative
# offset 0x4D in every Brahmic block we handle. Add a script's virama
# here and the rest of the segmenter (nukta, matras, modifiers, conjunct
# chaining) works for it automatically because those are detected by
# Unicode general category, not by script-specific code-point tables.
VIRAMAS = {
    "्",  # DEVANAGARI SIGN VIRAMA
    "్",  # TELUGU SIGN VIRAMA
}

COMBINING_CATEGORIES = {"Mn", "Mc", "Me"}  # nonspacing / spacing-combining / enclosing marks

DEVANAGARI_RANGE = (0x0900, 0x097F)
TELUGU_RANGE = (0x0C00, 0x0C7F)

# Word extractor: a "word" is a maximal run of letters, marks (matras,
# virama, nukta, modifiers all live in \p{M}), digits, or ZWJ/ZWNJ.
# Using \p{L}\p{M} instead of the ASCII-biased \w keeps virama/matra
# sequences glued to their base letter *before* akshara segmentation
# ever runs, so a naive whitespace/regex split can't sever a cluster.
_WORD_RE = regex.compile(r"[\p{L}\p{M}\p{Nd}‌‍]+")


def extract_words(text: str) -> list:
    """Split raw prose into word strings, Unicode-cluster-safe."""
    return _WORD_RE.findall(text)


def _is_combining(ch: str) -> bool:
    return unicodedata.category(ch) in COMBINING_CATEGORIES


def _in_range(ch: str, rng) -> bool:
    return rng[0] <= ord(ch) <= rng[1]


def is_brahmic(word: str) -> bool:
    return any(
        _in_range(ch, DEVANAGARI_RANGE) or _in_range(ch, TELUGU_RANGE)
        for ch in word
    )


def segment_aksharas(word: str) -> list:
    """
    Segment a Brahmic-script word into akshara (grapheme-cluster) units.

    Algorithm (applies equally to Devanagari and Telugu, see module
    docstring for why): a new akshara starts at any non-combining,
    non-virama, non-joiner character. It then greedily absorbs:
      - nukta / matras / modifier signs (any Mn/Mc/Me character), and
      - virama [+ optional ZWJ/ZWNJ] + consonant chains (conjuncts),
        which re-open the absorption loop so multi-consonant clusters
        (e.g. Devanagari "क्ष", "स्त्र") stay in one akshara, and
      - a bare trailing virama (a "dead" consonant with no vowel, e.g.
        the final letter of Hindi "विद्युत्") is absorbed with nothing
        further to chain onto.
    """
    clusters = []
    i, n = 0, len(word)
    while i < n:
        cluster = [word[i]]
        i += 1
        while i < n:
            ch = word[i]
            if ch in VIRAMAS:
                j = i + 1
                pending = [ch]
                if j < n and word[j] in JOINERS:
                    pending.append(word[j])
                    j += 1
                if j < n and unicodedata.category(word[j]) == "Lo":
                    # Conjunct: absorb virama(+joiner)+consonant and keep
                    # scanning -- more virama/consonant chains, or matras,
                    # may still follow onto the same akshara.
                    cluster.extend(pending)
                    cluster.append(word[j])
                    i = j + 1
                    continue
                else:
                    # Explicit halant / dead consonant at a word/akshara edge.
                    cluster.extend(pending)
                    i = j
                    continue
            elif _is_combining(ch) or ch in JOINERS:
                cluster.append(ch)
                i += 1
                continue
            else:
                break
        clusters.append("".join(cluster))
    return clusters


def segment_word(word: str) -> list:
    """Dispatch to akshara segmentation for Brahmic words, else per-character."""
    if is_brahmic(word):
        return segment_aksharas(word)
    return list(word)


# Guaranteed base-vocab members for every non-word pretoken a real
# document is likely to contain: full ASCII punctuation (this is what
# Markdown syntax is built from -- # * _ ` [ ] ( ) > | etc.), the
# whitespace variants worth distinguishing, common "smart" typography,
# and a couple of script-specific marks (Hindi danda, Spanish inverted
# punctuation) seen in the four source corpora. Encoding never *depends*
# on a character being in this set -- `tokenize_full_text` + the
# passthrough in `BPETokenizer.encode_text` round-trip ANY character,
# known or not -- this set just means the common ones are legitimately
# priced into the declared 10,000-token vocabulary instead of being an
# undeclared runtime fallback.
PUNCT_WHITESPACE_SEED = set(
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"  # ASCII punctuation/symbols
    " \t\n\r"                              # whitespace
    "‘’“”–—…•"  # ‘ ’ “ ” – — … •
    "।॥"                          # । ॥  (Hindi danda / double danda)
    "¿¡"                          # ¿ ¡  (Spanish inverted punctuation)
)


def is_word_pretoken(s: str) -> bool:
    """True for the word-run pretokens `tokenize_full_text` produces (these
    go through segment_word + BPE); false for the single-character
    whitespace/punctuation pretokens it also produces (these pass through
    literally -- see bpe.py::BPETokenizer.encode_text)."""
    return bool(_WORD_RE.fullmatch(s))


def tokenize_full_text(text: str) -> list:
    """
    Split arbitrary running text -- not just "words" -- into an ordered
    list of pretokens that covers EVERY character, word or not, such
    that "".join(tokenize_full_text(text)) == text exactly.

    `extract_words` only ever returns the word-character runs matched by
    `_WORD_RE`; every character in between (spaces, apostrophes, commas,
    periods, Markdown punctuation, ...) is invisible to it. That's fine
    for building a word-frequency vocabulary (Module 4's job), but a
    tokenizer that only knows about word characters cannot encode, let
    alone decode, real prose -- an apostrophe in "India's" or the commas
    in "1,428,627,663" simply have nowhere to go. This function is the
    fix: it walks the text once, alternating word runs (returned as a
    single string, to be BPE-segmented/merged downstream same as
    before) with every other character taken one at a time (so encoding
    never has to guess how to group unknown punctuation -- each
    character is its own pretoken and always round-trips).
    """
    pretokens = []
    i, n = 0, len(text)
    while i < n:
        m = _WORD_RE.match(text, i)
        if m:
            pretokens.append(m.group(0))
            i = m.end()
        else:
            pretokens.append(text[i])
            i += 1
    return pretokens


if __name__ == "__main__":
    tests = [
        "क्षत्रिय",      # kShatriya: initial conjunct क्ष + त्रि + य
        "विद्युत्",       # vidyut: ends in a bare/dead consonant (trailing virama)
        "हैं",           # hain: base + matra + anusvara, one akshara
        "स्वतंत्रता",     # svatantrata: nested conjuncts + anusvara
        "తెలుగు",        # Telugu: "telugu" itself, matra-heavy
        "సంవత్సరం",       # Telugu: anusvara + conjunct + anusvara
    ]
    for w in tests:
        print(f"{w!r:20s} -> {segment_aksharas(w)}")
