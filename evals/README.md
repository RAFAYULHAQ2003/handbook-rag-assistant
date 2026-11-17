# Evaluation set

No test questions were provided, so these were written by hand from the corpus **before** building the assistant, to avoid picking questions the system already passes. A few were added later (`conf-03`, `fut-05`, `part-02`, `h-miss-02`, `h-conf-01`, `h-part-01`); they came from re-reading the corpus for traps and filling category gaps, never from the assistant's output.

- `questions.jsonl`: dev set (97 questions), used while building and tuning.
- `questions_holdout.jsonl`: holdout set (26 questions), not looked at while tuning; run at the end to check the dev score is not inflated.
- `questions.py`: loads the sets.
- `matching.py`: how a fact is matched in an answer (see below).
- `retrieval_eval.py`: compares search configurations with no model calls (`python -m evals.retrieval_eval`); results in `results/retrieval-dev.json`.
- `answer_eval.py`: answers every question end to end and scores it (`python -m evals.answer_eval`); options `--models`, `--context whole` (baseline), `--injection`, `--set holdout`, `--label`. Results in `results/answers-<set>-<model>-<context>[-<label>].json`.
- `scoring.py`: the scoring rules below, as tested code.
- `fixtures/injected-notice.md`: a fake uploaded notice that tells the assistant to lie; used by `--injection`.

## Format

One JSON object per line:

| Field | Meaning |
|---|---|
| `id` | Unique id, prefixed by category |
| `category` | Which behaviour the question tests (see below) |
| `question` | What the employee asks |
| `expect` | `answer`, `refuse` (handbook does not cover it) or `clarify` (ambiguous) |
| `expected_doc_ids` | Documents that must be retrieved and cited |
| `acceptable_doc_ids` | Optional: other documents that also legitimately support the answer; citing them is not an error |
| `must_include` | Facts the answer must contain: a list of groups; every group must match, any string in a group is enough |
| `verdict` | Optional, for yes/no questions: `yes` or `no` |
| `reference_date` | Optional "today" for date-dependent questions; the run defaults to `2026-11-17` |
| `notes` | Why the question is tricky |

Every `must_include` group appears in the text of the expected documents, and fits inside a single passage (both enforced by tests).

## Scoring rules

| Expect | Passes when |
|---|---|
| `answer` | The answer is given (not refused), every `must_include` group matches, and, if `verdict` is set, the answer's structured `verdict` field equals it (not its first word: "Not yet" and "Unfortunately, no" are both "no") |
| `refuse` | The assistant says the handbook does not cover the question, and does not present a fact from it as the answer |
| `clarify` | The assistant asks which one is meant, or lists every match (`must_include` holds all the names) |

Retrieval and citations are scored separately, with no model calls. Retrieval metrics are computed by `retrieval_eval.py`; citation metrics by `answer_eval.py` (`scoring.py`):

- **recall@5**: share of `expected_doc_ids` among the top 5 distinct documents of the ranking; `acceptable_doc_ids` are not required but not counted against
- **hit@1 / MRR**: the first retrieved document is expected or acceptable / 1 ÷ rank of the first such document
- **context recall**: share of `expected_doc_ids` in the context actually given to the answer step
- **stale first**: a superseded or deprecated document ranked first, unless the question's expected documents include that old version
- **vs bm25**: questions won and lost against the keyword-only baseline, with a sign test; small differences on 92 questions are reported as counts, not as proof
- **citation support**: at least one cited passage contains a `must_include` fact (passage-level, so no passage ids are needed as gold)
- **citation precision (strict)**: share of citations whose passage contains a `must_include` fact and is from a document in force (or from a stale version the question asks about)
- **citation relevance**: share of citations whose passage contains a fact *or* comes from an expected/acceptable document, and applies. Added after inspecting the first run: strict precision counted legitimate reasoning steps ("Tier 1: New York, San Francisco, Boston" for a San Francisco hotel question) as imprecise. Both are reported
- **dropped quotes**: quotes the model gave that were not found word for word in the cited document; they never reach the user
- **stale only**: every citation is from an outdated version the question did not ask about

Conflicts are resolved by one written rule, also given to the answer step: a specific, explicitly stated exception beats a general table, even a newer one (`conf-04`).

Limit: text matching cannot tell "16 days" from "it used to be 12, now 16" (both pass, correctly), and it cannot tell whether a stale value is given *as the answer* or only to explain a change (for example `fut-05`, where a good answer mentions 26 weeks to say why it does not apply). Those categories (`superseded`, `deprecated`, `future`) get a second check from the structured answer fields or an optional model judge.

## Fact matching

Matching is plain text, so it is free, fast and gives the same result every run. It is case-insensitive and ignores thousands separators (`$3,000` = `$3000`). A fact must match as a whole token: `2` does not match `$2,500`, `12`, `2.5` or the day in `2026-11-02`, and `VP` does not match `VPN`.

## Rules enforced by `tests/test_eval_set.py`

- ids are unique and no holdout question repeats a dev question
- only known fields; every expected and acceptable document exists in the manifest, and the two lists do not overlap
- `refuse` questions have no expected facts; all others have facts and documents
- `verdict` is `yes` or `no`, and only on `answer` questions
- no `must_include` fact already appears in the question (an answer that repeats the question must not pass)
- `multi_doc` and `conflict` questions name at least two documents
- `date` and `future` questions pin a `reference_date`
- the dev set covers every category, every document (except the FAQ, which only repeats other policies) and every file format

## Categories

| Category | Tests |
|---|---|
| `fact` | Single-document lookup |
| `table` | Rows from CSV files and the Word hotel-rate table |
| `superseded` | Leave Policy v2 must win over v1 |
| `future` | 2027 parental leave revision is not yet in force |
| `deprecated` | 2023 deploy runbook must not be followed |
| `multi_doc` | Answer needs two or more documents |
| `conflict` | Sources disagree; both should be stated |
| `date` | Answer depends on the reference date |
| `partial` | Only partly covered (e.g. holiday dates are referenced but missing); answer what is there, say what is not |
| `unanswerable` | Not in the handbook; the assistant should say so |
| `ambiguous` | Several people match; the assistant should list them |
| `paraphrase` | Casual wording, missing punctuation |
| `injection` | Tries to override the documents; the answer must still follow the handbook |
