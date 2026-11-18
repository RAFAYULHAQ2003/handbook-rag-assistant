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

**Upload a document** (the paperclip) adds a `.md`, `.txt`, `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.csv`, `.json` or `.html` file of up to 5 MB **to the current conversation only**, and you can ask about it at once. The file is parsed, chunked and embedded for that conversation alone: it gets its own temporary ChromaDB collection and is never added to the handbook index or seen by anyone else. Everything of it (file, chunks and vectors) is deleted on "New conversation", after 60 minutes without use (`UPLOAD_SESSION_TTL_MINUTES`), and when the server restarts. Answers cite uploads like any document, but they are always marked "uploaded, not official": an upload never replaces or amends a handbook policy, and when it disagrees, the answer gives the official rule. Scanned PDFs (images of text) need OCR first and are refused with that explanation.

API (no login, JSON):

```bash
curl -X POST http://127.0.0.1:8000/api/ask -H "Content-Type: application/json" \
     -d '{"question": "How many days of annual leave do I get?"}'
curl http://127.0.0.1:8000/api/health
```

**Typos:** a question word that is neither a handbook word nor a known English word is also searched as the closest handbook word ("anual leves" → "annual leave"), using SymSpell (`symspellpy`) over the handbook's vocabulary, with its 82,000-word English dictionary keeping real words such as "dental" unchanged. The original words are kept, the model sees the question as typed, and the answer says what was also searched for.

**Repeated questions** come back instantly from the answer cache, whichever model (main or backup) gave the answer first.

**Vectors:** passages are embedded locally with `BAAI/bge-small-en-v1.5` and stored in ChromaDB (one collection per embedding model). Every web worker searches the same collection; uploads are added to it and removed from it at once. Keyword search (BM25), rank fusion and the document-state rules stay in the app. If ChromaDB cannot be reached, search uses an in-memory copy of the vectors and `/api/health` says so. Through ChromaDB, all 132 eval questions get exactly the same context as with in-memory search.

`/api/ask/stream` returns the same answer as newline-delimited JSON: first which documents are being read, then the answer. Add `"as_of": "2027-01-15"` to answer as of another date, and `"history": [{"question": "...", "answer": "..."}]` for a follow-up.

Uploads change state, so they need the CSRF token the page sets: `POST /api/upload` (multipart, field `file`), `GET /api/uploads`, and `POST /api/uploads/<doc_id>/delete`.

## Deployment (Google Cloud VM + Docker Compose, CI/CD with GitHub Actions)

Live: https://136-115-111-206.sslip.io

| Stage | Where | What |
|---|---|---|
| Test | GitHub Actions (`.github/workflows/ci.yml`), every push and pull request | lint and the full test suite |
| Build | GitHub Actions, `main` only, after the tests pass | builds the `Dockerfile` image for linux/amd64, stamped with the commit (`APP_VERSION`), and pushes it to Artifact Registry as `:<commit>` and `:latest` |
| Deploy | the VM (`deploy/vm/autodeploy.sh`, run every minute by a systemd timer) | updates the checkout to `main` (compose file, Caddyfile), pulls the new image, restarts, waits for `/api/health`, and rolls back to the previous image if it is not healthy |
| Verify | GitHub Actions | waits until `/api/health` on the live site reports the new commit |

GitHub signs in to Google Cloud with Workload Identity Federation. No key is stored anywhere: Google trusts only this repository's `main` branch, and the `github-deploy` service account may only push images to the `handbook` repository. Nothing reaches into the server from outside: GitHub pushes the image, and the VM pulls it.

The server is a Compute Engine e2-medium VM (Ubuntu 24.04) with a reserved IP. It runs `docker compose --profile https` (ChromaDB, the app, and Caddy with a Let's Encrypt certificate for `<ip-with-dashes>.sslip.io`). Its `.env` (debug off, its own secret key, `DOMAIN`, `WEB_IMAGE`) lives only on the server.

One-time server setup: clone the repository, create `.env` from the server lines in `.env.example`, then run `sudo sh deploy/vm/install.sh`. Deploy logs: `journalctl -u handbook-autodeploy`.
