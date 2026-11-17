import re

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its my of on or our "
    "should so than that the their them then there these they this to was we what when where which "
    "who whom why will with you your".split()
)


def _stem(token: str) -> str:
    """Fold plurals only ("policies" -> "policy", "days" -> "day"); heavier stemming hurts names and codes."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")) and token[-2].isalpha():
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens for keyword search.

    "$2,500" -> "2500"; compound tokens are kept whole and also split, so
    "x-request-id" matches both "x-request-id" and "request id", and "sev2" stays
    one token.
    """
    tokens: list[str] = []
    for match in _TOKEN.findall(_THOUSANDS.sub("", text.lower())):
        parts = re.split(r"[-.]", match)
        for t in ([match] + parts if len(parts) > 1 else [match]):
            if t and t not in STOPWORDS:
                tokens.append(_stem(t))
    return tokens
