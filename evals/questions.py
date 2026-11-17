"""Load and describe the evaluation question sets."""

import json
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
QUESTION_SETS = {
    "dev": EVALS_DIR / "questions.jsonl",
    "holdout": EVALS_DIR / "questions_holdout.jsonl",
}
DEFAULT_REFERENCE_DATE = "2026-11-17"

CATEGORIES = {
    "fact", "table", "superseded", "future", "deprecated", "multi_doc", "conflict",
    "date", "partial", "unanswerable", "ambiguous", "paraphrase", "injection",
}
EXPECTS = {"answer", "refuse", "clarify"}
VERDICTS = {"yes", "no"}
REQUIRED = {"id", "category", "question", "expect", "expected_doc_ids", "must_include"}
OPTIONAL = {"acceptable_doc_ids", "verdict", "reference_date", "notes"}


def load_questions(name_or_path: str | Path) -> list[dict]:
    """A question set by name ("dev", "holdout") or by path to a .jsonl file."""
    path = QUESTION_SETS.get(str(name_or_path), Path(name_or_path))
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def all_questions() -> list[tuple[str, dict]]:
    """(set name, question) for every question in every set."""
    return [(name, q) for name in QUESTION_SETS for q in load_questions(name)]


def reference_date_for(question: dict) -> str:
    return question.get("reference_date", DEFAULT_REFERENCE_DATE)
