"""Deterministic fact matching used to score answers.

A fact matches only as a whole token, so "2" does not match "$2,500", "2026"
or "12", and "VP" does not match "VPN". Thousands separators are ignored, so
"$3,000" and "$3000" are the same.
"""

import re

# every dash-like character counts as "-": models write U+2011 (non-breaking hyphen) in "6\u2011month"
_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
                         "\u2212": "-", "\u2019": "'", "\u00a0": " ", "\u202f": " "})


def normalize(text: str) -> str:
    text = text.translate(_DASHES).lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 3,000 -> 3000
    text = re.sub(r"[*_`]", "", text)  # markdown emphasis
    return re.sub(r"\s+", " ", text).strip()


def _pattern(alternative: str) -> re.Pattern:
    alt = normalize(alternative)
    body = re.escape(alt)
    prefix = suffix = ""
    if alt[0].isalnum():
        prefix = r"(?<![a-z0-9])"
        if alt[0].isdigit():
            # not the tail of a decimal or of an ISO date (2026-11-16)
            prefix += r"(?<!\d\.)(?<!\d{4}-)(?<!\d{4}-\d{2}-)"
    if alt[-1].isalnum():
        suffix = r"(?![a-z0-9])"
        if alt[-1].isdigit():
            suffix += r"(?!\.\d)"
    return re.compile(prefix + body + suffix)


def contains_fact(text: str, alternatives: list[str]) -> bool:
    """True if any alternative appears in text as a whole token."""
    norm = normalize(text)
    return any(_pattern(a).search(norm) for a in alternatives)


def facts_matched(text: str, must_include: list[list[str]]) -> list[bool]:
    """One bool per fact group: did the text contain that fact?"""
    return [contains_fact(text, group) for group in must_include]
