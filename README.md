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
python -m evals.answer_eval --set followup          # two-turn questions ("and who approves it?")
```

Answers are cached in `.cache/llm/`, so re-running unchanged questions makes no model calls.

## Run the app

Put a model key in `.env` first (`LLM_API_KEY`, with `LLM_BASE_URL` and `LLM_MODEL`; see `.env.example`).

**With Docker (the app plus its ChromaDB vector store):**

```bash
docker compose up --build
```

Open http://localhost:8000. The first build downloads the embedding model and embeds the handbook into the image, so the container starts in seconds. ChromaDB keeps its data in a Docker volume and is reachable only from this machine (`127.0.0.1:8001`).

**Without Docker:** start ChromaDB on its own (`docker compose up -d chroma`), or set `VECTOR_STORE=memory` to keep the vectors in memory. Then:

```bash
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000. Ask a question; each answer lists its sources, with `[1]`-style markers in the text pointing to the source behind each claim. Opening a source shows the document with the cited passage highlighted, plus a link to the original file. Follow-up questions ("and who approves it?") use the last three turns. A link like `/?q=How+many+days+of+annual+leave+do+I+get%3F` asks straight away.

**Upload a document** (below the question box) adds a `.md`, `.txt`, `.pdf`, `.docx`, `.csv` or `.html` file of up to 5 MB, and you can ask about it at once, without a restart. Uploads are stored in `media/uploads/` (set `UPLOAD_DIR` to change it), apart from the handbook. Answers cite them like any document, but they are always marked "uploaded, not official": they never replace or amend a handbook policy, and when they disagree, the answer gives the official rule. Remove one with the × on its chip.

API (no login, JSON):

```bash
curl -X POST http://127.0.0.1:8000/api/ask -H "Content-Type: application/json" \
     -d '{"question": "How many days of annual leave do I get?"}'
curl http://127.0.0.1:8000/api/health
```

**Vectors:** passages are embedded locally with `BAAI/bge-small-en-v1.5` and stored in ChromaDB (one collection per embedding model). Every web worker searches the same collection; uploads are added to it and removed from it at once. Keyword search (BM25), rank fusion and the document-state rules stay in the app. If ChromaDB cannot be reached, search uses an in-memory copy of the vectors and `/api/health` says so. Through ChromaDB, all 132 eval questions get exactly the same context as with in-memory search.

**Deploy on a server** with Docker: point a domain at it, or use `<ip-with-dashes>.sslip.io`. Set the server lines at the end of `.env.example`, then run:

```bash
docker compose --profile https up -d --build
```

Caddy gets and renews the HTTPS certificate. The app sits behind it, and ChromaDB is never exposed. Uploads, ChromaDB data and cached answers live in Docker volumes, so they survive restarts and rebuilds.

`/api/ask/stream` returns the same answer as newline-delimited JSON: first which documents are being read, then the answer. Add `"as_of": "2027-01-15"` to answer as of another date, and `"history": [{"question": "...", "answer": "..."}]` for a follow-up.

Uploads change state, so they need the CSRF token the page sets: `POST /api/upload` (multipart, field `file`), `GET /api/uploads`, and `POST /api/uploads/<doc_id>/delete`.
