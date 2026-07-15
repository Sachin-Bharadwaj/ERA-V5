"""
Module 3: From-scratch BPE trainer + encoder.

Operates over sequences of *atomic units* handed to it by segmenters.py
(aksharas for Brahmic words, characters for Latin words) -- never over
raw bytes or raw codepoints directly. That is the crucial difference
from a naive byte-level BPE (e.g. GPT-2 style): the merge algorithm
itself is generic, but it is only ever allowed to fuse whole atomic
units together, so a merge can never land inside a grapheme cluster.

Classic Sennrich et al. (2016) word-level BPE:
  - start from per-word symbol sequences (already unit-segmented)
  - repeatedly find the most frequent adjacent symbol pair (weighted by
    word frequency) and merge it into a new symbol
  - record merges in the order learned -- that order is the priority
    used later to encode unseen words
"""
from collections import Counter, defaultdict


class BPETokenizer:
    def __init__(self, name: str = ""):
        self.name = name
        self.merges = []          # ordered list of (a, b) -> merged "a+b"
        self.merge_rank = {}      # (a, b) -> priority (lower = learned earlier)
        self.vocab = set()        # every symbol ever produced (base units + merges)

    # ---- training -----------------------------------------------------

    def init_vocab(self, word_freq: dict, segment_fn) -> dict:
        """
        word_freq: {word_string: count}
        segment_fn: word_string -> list[atomic units]
        Returns {word_string: [symbol, symbol, ...]} the mutable working state.
        """
        state = {}
        for word in word_freq:
            symbols = segment_fn(word)
            state[word] = symbols
            self.vocab.update(symbols)
        return state

    @staticmethod
    def _count_pairs(state: dict, word_freq: dict) -> Counter:
        pairs = Counter()
        for word, symbols in state.items():
            freq = word_freq[word]
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        return pairs

    @staticmethod
    def _merge_symbol(symbols: list, pair: tuple) -> list:
        a, b = pair
        merged = a + b
        out = []
        i = 0
        n = len(symbols)
        while i < n:
            if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out

    def train_step(self, state: dict, word_freq: dict) -> str:
        """Perform exactly one merge (the globally most frequent pair). Returns
        the new merged symbol, or None if no pair is left to merge."""
        pairs = self._count_pairs(state, word_freq)
        if not pairs:
            return None
        best_pair, _ = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        a, b = best_pair
        merged = a + b
        for word in state:
            if a in state[word] and b in state[word]:
                state[word] = self._merge_symbol(state[word], best_pair)
        self.merge_rank[best_pair] = len(self.merges)
        self.merges.append(best_pair)
        self.vocab.add(merged)
        return merged

    def train(self, word_freq: dict, segment_fn, num_merges: int) -> dict:
        state = self.init_vocab(word_freq, segment_fn)
        for _ in range(num_merges):
            if self.train_step(state, word_freq) is None:
                break
        return state

    # ---- encoding -------------------------------------------------------

    def encode(self, symbols: list) -> list:
        """Apply learned merges, in learned order, to a fresh symbol sequence."""
        symbols = list(symbols)
        while len(symbols) > 1:
            pairs = list(zip(symbols, symbols[1:]))
            ranked = [(self.merge_rank[p], p) for p in pairs if p in self.merge_rank]
            if not ranked:
                break
            _, best_pair = min(ranked)
            symbols = self._merge_symbol(symbols, best_pair)
        return symbols

    def token_count(self, segment_fn, word: str) -> int:
        return len(self.encode(segment_fn(word)))

    def vocab_size(self) -> int:
        return len(self.vocab)

    # ---- full-text encode/decode ----------------------------------------
    #
    # `encode` above only ever handles a single pre-segmented word. Real
    # documents are not one word -- they're words interleaved with
    # whitespace and punctuation, and every one of those characters has
    # to survive a roundtrip. `encode_text` is the actual text-level API:
    # word-run pretokens go through segment_fn + the learned merges (same
    # as `encode`); every other pretoken (always exactly one character --
    # see `tokenize_full_text`) is passed through literally, so it is
    # never at risk of being dropped or replaced by an UNK placeholder,
    # whether or not that particular character was in the declared
    # vocabulary. `decode` is the exact inverse: BPE tokens are always
    # literal substrings of the input, so plain concatenation, in order,
    # reconstructs the original text exactly.

    def encode_text(self, text: str, segment_fn, tokenize_fn, is_word_fn) -> list:
        tokens = []
        for pretoken in tokenize_fn(text):
            if is_word_fn(pretoken):
                tokens.extend(self.encode(segment_fn(pretoken)))
            else:
                tokens.append(pretoken)
        return tokens

    @staticmethod
    def decode(tokens: list) -> str:
        return "".join(tokens)
