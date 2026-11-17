"""Typo-tolerant search: corrections are added to the search, real words are never changed."""

from datetime import date

import pytest

from rag.index import HandbookIndex
from rag.retrieval.search import ReferenceFinder, _title_phrases
from rag.retrieval.spelling import SpellFixer, known_english_words


@pytest.fixture(scope="module")
def fixer(docs):
    return SpellFixer(d.text for d in docs.values())


def test_a_swapped_pair_of_letters_counts_as_one_typo(fixer):
    assert fixer.correct("aproves") == "approve"  # one missing letter, via the singular form
    assert fixer.correct("expierd") == "expired"  # two letters swapped: one edit


@pytest.mark.parametrize("question, expected", [
    ("ohw mny anual leves do i get?", [("anual", "annual"), ("leves", "leave")]),
    ("my lapotp got stolen", [("lapotp", "laptop")]),
    ("vpn sertificate expierd", [("sertificate", "certificate"), ("expierd", "expired")]),
    ("corprate card montly limit", [("corprate", "corporate"), ("montly", "monthly")]),
])
def test_typos_are_corrected_to_handbook_words(fixer, question, expected):
    assert fixer.corrections(question) == expected


@pytest.mark.parametrize("question", [
    "Who is our dental insurance provider?",  # "dental" is real English: never turned into "rental"
    "Can I work from Portugal for a month?",
    "What will the weather be in Austin tomorrow?",
    "How many days of annual leave do I get?",
    "What is the Wi-Fi password?",  # too short to guess at
])
def test_real_words_and_correct_questions_are_left_alone(fixer, question):
    assert fixer.corrections(question) == []
    assert fixer.expand(question) == question


def test_corrections_are_added_never_replacing_what_was_typed(fixer):
    assert fixer.expand("anual leave") == "anual leave annual"


def test_known_english_words_come_from_the_symspell_dictionary():
    words = known_english_words()
    assert len(words) > 80000 and {"dental", "portugal", "weather", "endpoints"} <= words and "anual" not in words


def test_misspelled_question_finds_the_leave_policy(docs):
    index = HandbookIndex.build(list(docs.values()), date(2026, 11, 17), mode="bm25")
    retrieval = index.retrieve("ohw mny anual leves do i get?")
    assert retrieval.hits[0].passage.doc_id == "hr-leave-policy-v2"
    assert retrieval.corrections == [("anual", "annual"), ("leves", "leave")]
    assert retrieval.question == "ohw mny anual leves do i get?"  # the model sees what was typed


def test_uploaded_words_join_the_vocabulary(docs):
    from rag.models import Document, DocumentMeta
    upload = Document(DocumentMeta("upload-x", "Offsite", "x.md", "md", uploaded=True), "The Lisbon offsite agenda.")
    index = HandbookIndex.build(list(docs.values()) + [upload], date(2026, 11, 17), mode="bm25")
    assert index.spelling.corrections("lisbn offsite") == [("lisbn", "lisbon")]


# ---------- naming a document without its version ----------

@pytest.mark.parametrize("title, phrase", [
    ("Leave Policy v2", "Leave Policy"),
    ("Production Deploy Runbook (2023)", "Production Deploy Runbook"),
    ("Parental Leave Policy (2027 revision)", "Parental Leave Policy"),
])
def test_titles_also_match_without_their_version(title, phrase):
    assert phrase in _title_phrases(title)


def test_meeting_note_dates_are_not_version_markers():
    assert _title_phrases("Platform Meeting Notes, March 2026") == {"Platform Meeting Notes, March 2026"}


def test_see_the_leave_policy_finds_it_and_the_longest_name_wins(docs):
    finder = ReferenceFinder(list(docs.values()))
    assert set(finder.find("See the Leave Policy for entitlements")) == {"hr-leave-policy-v1", "hr-leave-policy-v2"}
    parental = set(finder.find("See the Parental Leave Policy for details"))
    assert "hr-leave-policy-v2" not in parental and "hr-parental-leave-policy" in parental


def test_only_the_version_in_force_is_brought_in(docs):
    index = HandbookIndex.build(list(docs.values()), date(2026, 11, 17), mode="bm25")
    context = {c.doc_id: c.reason for c in index.retrieve("Where do I request leave?").context}
    assert "hr-leave-policy-v2" in context and "hr-leave-policy-v1" not in context
