"""Typo-tolerant search: "ohw mny anual leves do i get?" also searches for "annual leave".

Keyword search matches exact words, so one typo can hide the right document. Each question word
that is neither a handbook word nor a known English word is matched to the closest handbook word
(one typo for short words, two for long ones; swapping two letters counts as one), and the
correction is ADDED to the search text. Nothing the person typed is removed, and the model always
sees the question exactly as typed.

Built on SymSpell (symspellpy, MIT): its fast lookup searches the handbook's own vocabulary, and its
bundled 82,000-word English frequency dictionary is the guard that keeps real words the handbook
never uses ("dental", "Portugal") from being "corrected" into handbook words ("rental").
"""

import re
from collections import Counter
from functools import lru_cache
from importlib.resources import files

from symspellpy import SymSpell, Verbosity

from rag.retrieval.text import STOPWORDS, _stem

_WORD = re.compile(r"[a-z]+")
MIN_LENGTH = 4  # shorter words are too ambiguous to correct ("mny": many, may, my, any?)
MAX_EDITS = 2


@lru_cache(maxsize=1)
def known_english_words() -> frozenset[str]:
    """Words of the English frequency dictionary that ships with symspellpy."""
    text = (files("symspellpy") / "frequency_dictionary_en_82_765.txt").read_text(encoding="utf-8")
    return frozenset(line.split(maxsplit=1)[0] for line in text.splitlines() if line.strip())


class SpellFixer:
    """Corrects question words against the vocabulary of the documents being searched."""

    def __init__(self, texts, known: frozenset[str] | None = None):
        self.counts = Counter(w for text in texts for w in _WORD.findall(text.lower()))
        self.known = known if known is not None else known_english_words()
        self._symspell = SymSpell(max_dictionary_edit_distance=MAX_EDITS, prefix_length=7)
        for word, count in self.counts.items():
            if len(word) >= 3:
                self._symspell.create_dictionary_entry(word, count)

    def correct(self, word: str) -> str | None:
        """The handbook word this one is a typo of, or None."""
        if len(word) < MIN_LENGTH or word in self.counts or word in self.known or word in STOPWORDS:
            return None
        limit = 1 if len(word) <= 6 else MAX_EDITS
        best = None
        # also try the word without a plural ending: "leves" -> "leve" is one typo from "leave"
        for form in dict.fromkeys((word, _stem(word))):
            for s in self._symspell.lookup(form, Verbosity.CLOSEST, max_edit_distance=limit):
                # fewest edits, then same first letter, then the most common handbook word
                key = (s.distance, s.term[0] != word[0], -s.count, s.term)
                if best is None or key < best[0]:
                    best = (key, s.term)
        return best[1] if best else None

    def corrections(self, question: str) -> list[tuple[str, str]]:
        """(typed, corrected) pairs, in question order, each typed word once."""
        out, seen = [], set()
        for word in _WORD.findall(question.lower()):
            if word in seen:
                continue
            seen.add(word)
            fixed = self.correct(word)
            if fixed:
                out.append((word, fixed))
        return out

    def expand(self, question: str) -> str:
        """The search text: the question as typed, plus any corrected words."""
        fixes = self.corrections(question)
        return f"{question} {' '.join(f for _, f in fixes)}" if fixes else question
