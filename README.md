# Handbook Q&A Assistant

An assistant that answers Northgate Robotics employees' questions from the company handbook, citing the document and passage each answer rests on.

> Work in progress.

## Project layout

| Path | What lives there |
|---|---|
| `rag/` | Core logic in plain Python (no Django): loading, validity rules, chunking, retrieval, answering |
| `webapp/` | Django project; `webapp/handbook/` is the app (pages, API, upload) |
| `evals/` | Evaluation questions, metrics and saved results |
| `tests/` | Unit and web tests |
| `data/corpus/` | The handbook documents and `manifest.json` (never modified) |

## Setup

Requires **Python 3.12 or newer** (Django 6.1 needs it).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run tests

```bash
pytest
```

The first run of the app downloads the local embedding model (about 70 MB) into `.cache/models`. Until it is downloaded, the hybrid-search quality test is skipped and search falls back to keyword-only (BM25).

## Evaluate

```bash
python -m evals.retrieval_eval                      # search configurations on the dev set, no model calls
python -m evals.answer_eval --label thinking-on     # every dev question answered and scored (uses LLM_* in .env)
python -m evals.answer_eval --context whole         # baseline: send the whole handbook instead of searching
python -m evals.answer_eval --injection             # an uploaded document tries to make the assistant lie
python -m evals.answer_eval --set holdout           # the blind set; run once, at the end
```

Answers are cached in `.cache/llm/`, so re-running unchanged questions makes no model calls.

## Run the app

```bash
python manage.py migrate
python manage.py runserver
```
