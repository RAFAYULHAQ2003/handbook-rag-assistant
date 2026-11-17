import re
from dataclasses import dataclass
from datetime import date
from enum import Enum

from rag.models import DocumentMeta


class State(str, Enum):
    IN_FORCE = "in_force"
    SUPERSEDED = "superseded"
    DEPRECATED = "deprecated"
    NOT_YET_EFFECTIVE = "not_yet_effective"


class Authority(str, Enum):
    POLICY = "policy"
    INFORMAL = "informal"


DEPRECATED_STATUSES = {"deprecated", "archived", "obsolete", "retired", "withdrawn"}
INFORMAL_CATEGORIES = {"meeting notes", "notes"}


@dataclass(frozen=True)
class DocStatus:
    doc_id: str
    state: State
    authority: Authority
    reference_date: date
    effective_date: date | None
    family: str
    current_version: str | None = None  # the in-force document to use instead (stale or upcoming docs)
    superseded_by: str | None = None  # explicit `supersedes` successor that is in force
    upcoming: tuple[str, ...] = ()  # not-yet-effective documents that revise this one
    amended_by: tuple[str, ...] = ()  # newer in-force versions that amend (not replace) this one

    @property
    def in_force(self) -> bool:
        return self.state is State.IN_FORCE


@dataclass(frozen=True)
class ValidityFinding:
    doc_id: str
    issue: str
    detail: str


_PAREN = re.compile(r"\([^)]*\)")
_VERSION_WORDS = re.compile(r"\b(v\d+(\.\d+)*|version \d+|rev(ision)?|draft|(19|20)\d{2})\b")


def family_key(title: str) -> str:
    """Versions of one document share a key: 'Leave Policy v1', 'Leave Policy v2' -> 'leave policy'."""
    key = _PAREN.sub(" ", title.lower())
    key = _VERSION_WORDS.sub(" ", key)
    return re.sub(r"[^a-z0-9]+", " ", key).strip()


def _authority(meta: DocumentMeta) -> Authority:
    category = (meta.category or "").strip().lower()
    return Authority.INFORMAL if category in INFORMAL_CATEGORIES else Authority.POLICY


def assess(metas: list[DocumentMeta], ref: date) -> tuple[dict[str, DocStatus], list[ValidityFinding]]:
    """Compute the state of every document as of `ref`."""
    by_id = {m.doc_id: m for m in metas}
    findings: list[ValidityFinding] = []

    # explicit supersession edges, predecessor -> successors
    successors: dict[str, list[str]] = {}
    for m in metas:
        target = m.supersedes
        if not target:
            continue
        if target == m.doc_id:
            findings.append(ValidityFinding(m.doc_id, "supersedes_itself", target))
        elif target not in by_id:
            findings.append(ValidityFinding(m.doc_id, "supersedes_unknown_document", target))
        else:
            successors.setdefault(target, []).append(m.doc_id)

    def basic_state(m: DocumentMeta) -> State:
        if (m.status or "").lower() in DEPRECATED_STATUSES:
            return State.DEPRECATED
        if m.effective_date and m.effective_date > ref:
            return State.NOT_YET_EFFECTIVE
        return State.IN_FORCE

    basic = {m.doc_id: basic_state(m) for m in metas}

    def latest_in_force(doc_id: str, seen: frozenset = frozenset()) -> str | None:
        """Newest in-force document along the supersession chain after `doc_id`.

        Only in-force successors count; when several qualify, the most recently
        effective one wins.
        """
        candidates = []
        for s in successors.get(doc_id, []):
            if s in seen or s == doc_id:
                findings.append(ValidityFinding(doc_id, "supersession_cycle", s))
                continue
            if basic[s] is State.IN_FORCE:
                candidates.append(latest_in_force(s, seen | {doc_id}) or s)
        if not candidates:
            return None
        return max(candidates, key=lambda d: (by_id[d].effective_date or date.min, d))

    families: dict[str, list[str]] = {}
    for m in metas:
        families.setdefault(family_key(m.title), []).append(m.doc_id)

    def newest_in_force_in_family(m: DocumentMeta) -> str | None:
        candidates = [by_id[d] for d in families[family_key(m.title)]
                      if d != m.doc_id and basic[d] is State.IN_FORCE and d not in successors]
        candidates.sort(key=lambda c: c.effective_date or date.min, reverse=True)
        return candidates[0].doc_id if candidates else None

    statuses: dict[str, DocStatus] = {}
    for m in metas:
        state = basic[m.doc_id]
        superseded_by = latest_in_force(m.doc_id) if state is State.IN_FORCE else None
        if superseded_by:
            state = State.SUPERSEDED

        current_version = None
        if state is State.SUPERSEDED:
            current_version = superseded_by
        elif state is State.DEPRECATED:
            current_version = latest_in_force(m.doc_id) or newest_in_force_in_family(m)
        elif state is State.NOT_YET_EFFECTIVE:
            target = m.supersedes if m.supersedes in by_id else None
            current_version = target if target and basic[target] is State.IN_FORCE else newest_in_force_in_family(m)

        upcoming = tuple(sorted(
            d for d in set(successors.get(m.doc_id, [])) | set(families[family_key(m.title)])
            if d != m.doc_id and basic[d] is State.NOT_YET_EFFECTIVE
        )) if state is State.IN_FORCE else ()

        amended_by = tuple(sorted(
            d for d in families[family_key(m.title)]
            if d != m.doc_id and basic[d] is State.IN_FORCE and d not in successors.get(m.doc_id, [])
            and m.doc_id not in successors.get(d, [])
            and (by_id[d].effective_date or date.min) > (m.effective_date or date.min)
        )) if state is State.IN_FORCE else ()

        statuses[m.doc_id] = DocStatus(
            doc_id=m.doc_id,
            state=state,
            authority=_authority(m),
            reference_date=ref,
            effective_date=m.effective_date,
            family=family_key(m.title),
            current_version=current_version,
            superseded_by=superseded_by,
            upcoming=upcoming,
            amended_by=amended_by,
        )

        declared = (m.status or "").lower()
        if declared == "current" and state is not State.IN_FORCE:
            findings.append(ValidityFinding(m.doc_id, "declared_current_but_not_in_force", state.value))
        if m.effective_date is None:
            findings.append(ValidityFinding(m.doc_id, "no_effective_date", "treated as in force"))

    unique = list(dict.fromkeys(findings))
    return statuses, unique


def describe(status: DocStatus, titles: dict[str, str]) -> str:
    """One-line explanation for answers and the web page."""
    def title(doc_id):
        return titles.get(doc_id, doc_id)

    if status.state is State.SUPERSEDED:
        return f"Superseded by {title(status.superseded_by)}."
    if status.state is State.DEPRECATED:
        tail = f" Use {title(status.current_version)}." if status.current_version else ""
        return f"Deprecated: kept for reference only.{tail}"
    if status.state is State.NOT_YET_EFFECTIVE:
        tail = f" Until then, {title(status.current_version)} applies." if status.current_version else ""
        return f"Not yet in effect: takes effect on {status.effective_date.isoformat()}.{tail}"
    if status.amended_by:
        names = ", ".join(title(d) for d in status.amended_by)
        return f"In force, as amended by {names}; where they differ, the newer text applies."
    if status.upcoming:
        names = ", ".join(title(d) for d in status.upcoming)
        return f"In force. A revision is scheduled: {names}."
    if status.authority is Authority.INFORMAL:
        return "Informal meeting notes: check the official policy for binding rules."
    return "In force."
