# Evaluation set

No test questions were provided, so these were written by hand from the corpus **before** building the assistant, to avoid picking questions the system already passes.

- `questions.jsonl`: dev set (64 questions), used while building and tuning.
- `questions_holdout.jsonl`: holdout set (20 questions), not looked at while tuning; run at the end to check the dev score is not inflated.
- `matching.py`: how a fact is matched in an answer (see below).

## Format

One JSON object per line:

| Field | Meaning |
|---|---|
| `id` | Unique id, prefixed by category |
| `category` | Which behaviour the question tests (see below) |
| `question` | What the employee asks |
| `expect` | `answer`, `refuse` (handbook does not cover it) or `clarify` (ambiguous) |
| `expected_doc_ids` | Documents that should be retrieved and cited |
| `must_include` | Facts the answer must contain: a list of groups; every group must match, any string in a group is enough |
| `reference_date` | Optional "today" for date-dependent questions; the run defaults to `2026-11-17` |
| `notes` | Why the question is tricky |

Every `must_include` group was checked to appear in the text of the expected documents.

## Fact matching

Matching is plain text, so it is free, fast and gives the same result every run. It is case-insensitive and ignores thousands separators (`$3,000` = `$3000`). A fact must match as a whole token: `2` does not match `$2,500`, `12`, `2.5` or the day in `2026-11-02`, and `VP` does not match `VPN`.

## Rules enforced by `tests/test_eval_set.py`

- ids are unique and no holdout question repeats a dev question
- every expected document exists in the manifest
- `refuse` questions have no expected facts; all others have facts and documents
- `multi_doc` and `conflict` questions name at least two documents
- `date` and `future` questions pin a `reference_date`
- the dev set covers every category

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
