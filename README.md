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

## Evaluate search

```bash
python -m evals.retrieval_eval        # compares search configurations on the dev set, no model calls
```

## Run the app

```bash
python manage.py migrate
python manage.py runserver
```
