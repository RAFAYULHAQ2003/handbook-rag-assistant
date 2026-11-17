import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from evals.questions import all_questions
from rag.config import reference_date
from rag.models import DocumentMeta
from rag.validity import Authority, State, assess, describe, family_key

TODAY = date(2026, 11, 17)

ALL = all_questions()


def meta(doc_id, title=None, effective=None, status="current", supersedes=None, category=None):
    return DocumentMeta(doc_id=doc_id, title=title or doc_id, path=f"{doc_id}.md", format="md",
                        effective_date=effective, status=status, supersedes=supersedes, category=category)


@pytest.fixture(scope="module")
def corpus_metas(docs):
    return [d.meta for d in docs.values()]


@pytest.fixture(scope="module")
def today(corpus_metas):
    return assess(corpus_metas, TODAY)


# ---------- the real handbook, as of the eval date ----------

def test_leave_v1_superseded_by_v2(today):
    statuses, _ = today
    v1 = statuses["hr-leave-policy-v1"]
    assert v1.state is State.SUPERSEDED
    assert v1.superseded_by == v1.current_version == "hr-leave-policy-v2"
    assert statuses["hr-leave-policy-v2"].in_force


def test_2023_runbook_deprecated_and_points_to_current(today):
    statuses, _ = today
    old = statuses["eng-deploy-runbook-2023"]
    assert old.state is State.DEPRECATED
    assert old.current_version == "eng-deploy-runbook"


def test_2027_parental_leave_not_yet_effective(today):
    statuses, _ = today
    future = statuses["hr-parental-leave-policy-2027"]
    assert future.state is State.NOT_YET_EFFECTIVE
    assert future.current_version == "hr-parental-leave-policy"
    current = statuses["hr-parental-leave-policy"]
    assert current.in_force
    assert current.upcoming == ("hr-parental-leave-policy-2027",)


def test_meeting_notes_are_informal_but_in_force(today):
    statuses, _ = today
    notes = [s for s in statuses.values() if s.doc_id.startswith("notes-")]
    assert len(notes) == 3
    assert all(s.in_force and s.authority is Authority.INFORMAL for s in notes)
    assert statuses["hr-leave-policy-v2"].authority is Authority.POLICY


def test_everything_else_in_force(today):
    statuses, _ = today
    stale = {d for d, s in statuses.items() if not s.in_force}
    assert stale == {"hr-leave-policy-v1", "eng-deploy-runbook-2023", "hr-parental-leave-policy-2027"}


def test_declared_status_mismatches_reported(today):
    _, findings = today
    issues = {(f.doc_id, f.issue) for f in findings}
    assert ("hr-leave-policy-v1", "declared_current_but_not_in_force") in issues
    assert ("hr-parental-leave-policy-2027", "declared_current_but_not_in_force") in issues


def test_descriptions(today, docs):
    statuses, _ = today
    titles = {d: doc.meta.title for d, doc in docs.items()}
    assert describe(statuses["hr-leave-policy-v1"], titles) == "Superseded by Leave Policy v2."
    assert "Use Production Deploy Runbook." in describe(statuses["eng-deploy-runbook-2023"], titles)
    assert "Until then, Parental Leave Policy applies." in describe(
        statuses["hr-parental-leave-policy-2027"], titles)
    assert describe(statuses["hr-referral-policy"], titles) == "In force."


# ---------- the real handbook on other dates ----------

def test_from_2027_the_revision_amends_the_2024_policy(corpus_metas, docs):
    statuses, _ = assess(corpus_metas, date(2027, 1, 1))
    assert statuses["hr-parental-leave-policy-2027"].in_force
    base = statuses["hr-parental-leave-policy"]
    assert base.in_force  # an amendment, not a replacement: "all other terms continue"
    assert base.amended_by == ("hr-parental-leave-policy-2027",)
    titles = {d: doc.meta.title for d, doc in docs.items()}
    assert "the newer text applies" in describe(base, titles)


def test_day_before_v2_took_effect(corpus_metas):
    statuses, _ = assess(corpus_metas, date(2025, 3, 31))
    assert statuses["hr-leave-policy-v1"].in_force
    assert statuses["hr-leave-policy-v1"].upcoming == ("hr-leave-policy-v2",)
    assert statuses["hr-leave-policy-v2"].state is State.NOT_YET_EFFECTIVE
    assert statuses["hr-leave-policy-v2"].current_version == "hr-leave-policy-v1"


def test_effective_on_its_own_date(corpus_metas):
    statuses, _ = assess(corpus_metas, date(2025, 4, 1))
    assert statuses["hr-leave-policy-v2"].in_force
    assert statuses["hr-leave-policy-v1"].state is State.SUPERSEDED


# ---------- version families ----------

@pytest.mark.parametrize("a,b", [
    ("Leave Policy v1", "Leave Policy v2"),
    ("Parental Leave Policy", "Parental Leave Policy (2027 revision)"),
    ("Production Deploy Runbook", "Production Deploy Runbook (2023)"),
])
def test_versions_share_a_family(a, b):
    assert family_key(a) == family_key(b)


def test_different_documents_do_not_share_a_family():
    assert family_key("Leave Policy v2") != family_key("Parental Leave Policy")


def test_only_the_intended_families_exist_in_the_corpus(corpus_metas):
    groups = {}
    for m in corpus_metas:
        groups.setdefault(family_key(m.title), set()).add(m.doc_id)
    multi = sorted(sorted(g) for g in groups.values() if len(g) > 1)
    assert multi == [
        ["eng-deploy-runbook", "eng-deploy-runbook-2023"],
        ["hr-leave-policy-v1", "hr-leave-policy-v2"],
        ["hr-parental-leave-policy", "hr-parental-leave-policy-2027"],
    ]


# ---------- edge cases ----------

def test_chain_resolves_to_latest_in_force():
    metas = [meta("p1", effective=date(2020, 1, 1)),
             meta("p2", effective=date(2022, 1, 1), supersedes="p1"),
             meta("p3", effective=date(2024, 1, 1), supersedes="p2")]
    s, _ = assess(metas, TODAY)
    assert s["p1"].superseded_by == "p3"
    assert s["p2"].superseded_by == "p3"
    assert s["p3"].in_force


def test_future_successor_does_not_supersede_yet():
    metas = [meta("p1", effective=date(2020, 1, 1)),
             meta("p2", effective=date(2022, 1, 1), supersedes="p1"),
             meta("p3", effective=date(2030, 1, 1), supersedes="p2")]
    s, _ = assess(metas, TODAY)
    assert s["p1"].superseded_by == "p2"
    assert s["p2"].in_force and s["p2"].upcoming == ("p3",)
    assert s["p3"].current_version == "p2"


def test_deprecated_successor_does_not_supersede():
    metas = [meta("old", effective=date(2020, 1, 1)),
             meta("new", effective=date(2022, 1, 1), supersedes="old", status="withdrawn")]
    s, _ = assess(metas, TODAY)
    assert s["old"].in_force
    assert s["new"].state is State.DEPRECATED


def test_deprecated_wins_over_future_date():
    s, _ = assess([meta("x", effective=date(2030, 1, 1), status="Deprecated")], TODAY)
    assert s["x"].state is State.DEPRECATED


def test_cycle_does_not_crash():
    metas = [meta("a", effective=date(2020, 1, 1), supersedes="b"),
             meta("b", effective=date(2021, 1, 1), supersedes="a")]
    s, findings = assess(metas, TODAY)
    assert set(s) == {"a", "b"}
    assert any(f.issue == "supersession_cycle" for f in findings)


def test_self_and_unknown_supersedes_reported():
    metas = [meta("a", effective=date(2020, 1, 1), supersedes="a"),
             meta("b", effective=date(2020, 1, 1), supersedes="missing")]
    s, findings = assess(metas, TODAY)
    assert s["a"].in_force and s["b"].in_force
    assert {f.issue for f in findings} >= {"supersedes_itself", "supersedes_unknown_document"}


def test_missing_effective_date_treated_as_in_force():
    s, findings = assess([meta("upload")], TODAY)
    assert s["upload"].in_force
    assert any(f.issue == "no_effective_date" for f in findings)


def test_upload_superseding_a_corpus_document(corpus_metas):
    upload = meta("hr-leave-policy-v3", title="Leave Policy v3", effective=date(2026, 6, 1),
                  supersedes="hr-leave-policy-v2", category="HR")
    s, _ = assess(corpus_metas + [upload], TODAY)
    assert s["hr-leave-policy-v2"].superseded_by == "hr-leave-policy-v3"
    assert s["hr-leave-policy-v1"].superseded_by == "hr-leave-policy-v3"


# ---------- reference date ----------

def test_reference_date_sources(monkeypatch):
    monkeypatch.delenv("REFERENCE_DATE", raising=False)
    monkeypatch.delenv("COMPANY_TIMEZONE", raising=False)
    assert reference_date() == datetime.now(ZoneInfo("America/Chicago")).date()
    monkeypatch.setenv("REFERENCE_DATE", "2026-11-17")
    assert reference_date() == TODAY
    assert reference_date("2027-01-01") == date(2027, 1, 1)
    assert reference_date(date(2025, 1, 1)) == date(2025, 1, 1)


def test_reference_date_invalid(monkeypatch):
    monkeypatch.setenv("REFERENCE_DATE", "17/11/2026")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        reference_date()


# ---------- consistency with the eval set ----------

STALE_OK = {"superseded", "deprecated", "future"}


@pytest.mark.parametrize("source,q", ALL, ids=[q["id"] for _, q in ALL])
def test_eval_answers_rest_on_in_force_documents(corpus_metas, source, q):
    """Outside the stale-version categories, every expected document must be in force on the eval date."""
    if q["category"] in STALE_OK:
        return
    statuses, _ = assess(corpus_metas, date.fromisoformat(q.get("reference_date", "2026-11-17")))
    stale = [d for d in q["expected_doc_ids"] if not statuses[d].in_force]
    assert not stale, f"{q['id']} relies on documents not in force: {stale}"


def test_statuses_are_json_friendly(today):
    statuses, _ = today
    s = statuses["hr-leave-policy-v1"]
    assert json.dumps({"state": s.state, "authority": s.authority}) == \
        '{"state": "superseded", "authority": "policy"}'


def test_two_successors_newest_wins():
    metas = [meta("base", effective=date(2020, 1, 1)),
             meta("newer", effective=date(2024, 1, 1), supersedes="base"),
             meta("older", effective=date(2022, 1, 1), supersedes="base")]
    s, _ = assess(metas, TODAY)
    assert s["base"].superseded_by == "newer"


def test_amendment_link_points_only_forward(corpus_metas):
    statuses, _ = assess(corpus_metas, date(2027, 1, 1))
    assert statuses["hr-parental-leave-policy-2027"].amended_by == ()


def test_deprecated_points_to_an_in_force_version_not_another_old_one():
    metas = [meta("rb", title="Deploy Runbook", effective=date(2025, 1, 1)),
             meta("rb-2023", title="Deploy Runbook (2023)", effective=date(2023, 1, 1), status="deprecated"),
             # a withdrawn copy with a later date must not be chosen over the in-force version
             meta("rb-draft", title="Deploy Runbook (draft)", effective=date(2025, 6, 1), status="withdrawn")]
    s, _ = assess(metas, TODAY)
    assert s["rb-2023"].current_version == "rb"
    assert s["rb-draft"].current_version == "rb"



def test_withdrawn_middle_version_does_not_break_the_chain():
    metas = [meta("v1", effective=date(2020, 1, 1)),
             meta("v2", effective=date(2021, 1, 1), status="withdrawn", supersedes="v1"),
             meta("v3", effective=date(2022, 1, 1), supersedes="v2")]
    s, _ = assess(metas, TODAY)
    assert s["v1"].superseded_by == "v3"
    assert s["v2"].state is State.DEPRECATED and s["v2"].current_version == "v3"


def test_deprecated_still_points_to_a_version_with_only_a_future_successor():
    metas = [meta("rb", title="Runbook", effective=date(2025, 1, 1)),
             meta("rb-old", title="Runbook (2023)", effective=date(2023, 1, 1), status="deprecated"),
             meta("rb-next", title="Runbook (2027)", effective=date(2027, 6, 1), supersedes="rb")]
    s, _ = assess(metas, TODAY)
    assert s["rb"].in_force and s["rb"].upcoming == ("rb-next",)
    assert s["rb-old"].current_version == "rb"


def test_cycle_leaves_both_documents_in_force():
    metas = [meta("a", effective=date(2020, 1, 1), supersedes="b"),
             meta("b", effective=date(2021, 1, 1), supersedes="a")]
    s, findings = assess(metas, TODAY)
    assert s["a"].in_force and s["b"].in_force
    assert {(f.doc_id, f.issue) for f in findings} >= {("a", "supersession_cycle"), ("b", "supersession_cycle")}


def test_longer_cycle_does_not_affect_documents_outside_it():
    metas = [meta("a", supersedes="c"), meta("b", supersedes="a"), meta("c", supersedes="b"),
             meta("old"), meta("new", effective=date(2022, 1, 1), supersedes="old")]
    s, _ = assess(metas, TODAY)
    assert all(s[d].in_force for d in "abc")
    assert s["old"].superseded_by == "new"


def test_today_follows_the_company_time_zone_not_the_server(monkeypatch):
    monkeypatch.delenv("REFERENCE_DATE", raising=False)
    monkeypatch.setenv("COMPANY_TIMEZONE", "Pacific/Kiritimati")  # UTC+14 vs UTC-12: 26 hours apart, so always a different day
    kiritimati = reference_date()
    monkeypatch.setenv("COMPANY_TIMEZONE", "Etc/GMT+12")
    assert reference_date() != kiritimati
    assert kiritimati == datetime.now(ZoneInfo("Pacific/Kiritimati")).date()


def test_unknown_time_zone_rejected(monkeypatch):
    monkeypatch.delenv("REFERENCE_DATE", raising=False)
    monkeypatch.setenv("COMPANY_TIMEZONE", "Mars/Olympus")
    with pytest.raises(ValueError, match="COMPANY_TIMEZONE"):
        reference_date()


def test_an_upload_never_replaces_or_amends_an_official_document():
    official = meta("hr-leave-policy-v2", "Leave Policy v2", date(2026, 1, 1))
    upload = meta("upload-leave", "Leave Policy v3", date(2026, 6, 1), supersedes="hr-leave-policy-v2")
    upload.uploaded = True
    statuses, _ = assess([official, upload], date(2026, 11, 17))
    assert statuses["hr-leave-policy-v2"].state is State.IN_FORCE
    assert statuses["hr-leave-policy-v2"].amended_by == () and statuses["hr-leave-policy-v2"].upcoming == ()
    assert statuses["upload-leave"].state is State.IN_FORCE
