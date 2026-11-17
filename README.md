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

## Run the app

```bash
python manage.py migrate
python manage.py runserver
```
