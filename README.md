# Handbook Q&A Assistant

Ask questions about the Northgate Robotics employee handbook. Every answer shows the exact passages it comes from.

**Live demo:** https://136-115-111-206.sslip.io

---

## What it does

- Answers questions using only the 64 handbook documents (md, pdf, docx, csv, txt, html).
- Shows the source of every fact, with `[1]`-style markers and source cards.
- Opens any source with the cited passage highlighted, plus the original file.
- Knows which documents are current, outdated or not yet in force.
- Can answer "as of a date", for example a policy that starts in 2027.
- Understands follow-up questions ("and who approves it?").
- Tolerates typos ("ohw mny anual leves" finds the leave policy).
- Says "not covered" instead of guessing.
- Resists instructions hidden in questions or documents (prompt injection).
- Lets you upload a document and ask about it straight away (see [Uploads](#uploads)).

---

## Quick start (Docker)

You need Docker and an API key for an OpenAI-compatible model. We use NVIDIA's free API: get a key at https://build.nvidia.com. The model address and names are already set in `.env.example`.

```bash
git clone https://github.com/RAFAYULHAQ2003/handbook-rag-assistant.git
cd handbook-rag-assistant
cp .env.example .env          # then put your key in LLM_API_KEY
docker compose up --build
```

Open **http://localhost:8000**.

---

## Run without Docker

Needs **Python 3.12+**.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # put your key in LLM_API_KEY, and set VECTOR_STORE=memory
python manage.py migrate
python manage.py runserver
```

Open **http://127.0.0.1:8000**.

- The first run downloads a small embedding model (about 70 MB).
- `VECTOR_STORE=memory` keeps vectors in memory. To use ChromaDB instead, run `docker compose up -d chroma`.

---

## Settings (`.env`)

| Setting | What it is |
|---|---|
| `LLM_API_KEY` | Your model API key (**required**) |
| `LLM_BASE_URL` | Model API address, e.g. `https://integrate.api.nvidia.com/v1` |
| `LLM_MODEL` | Main model, e.g. `nvidia/nemotron-3-ultra-550b-a55b` |
| `LLM_FALLBACK_MODEL` | Backup model, used when the main one is slow or busy |
| `DJANGO_SECRET_KEY` | Any long random text |
| `DJANGO_DEBUG` | `1` locally, `0` on a server |
| `VECTOR_STORE` | `chroma` (default) or `memory` |

All settings, with comments, are in [`.env.example`](.env.example). Never commit `.env`.

---

## Try these questions

| Question | Expected answer |
|---|---|
| How many days of annual leave do I get? | 16 days a year (Leave Policy v2) |
| Can I deploy to production today? *(set the date to 17 Nov 2026)* | **No**: deploy freeze before release 2026.22 |
| What is the per-diem for an L1 employee visiting the Denver office? | $40 a day (Denver falls under Tier 3) |
| Does a $4,000 trip booked through VoyageHub need VP approval? | **No**: VoyageHub travel only needs the manager |
| Who is Maya? → which one is based in Austin? | Lists three Mayas → **Maya Brooks** |

---

## How it works

1. **Read:** every document is turned into clean text, then cut into passages.
2. **Check status:** each document is marked current, outdated or not yet in force for the chosen date.
3. **Search:** keyword search (BM25) plus meaning search (embeddings in ChromaDB) find the best passages.
4. **Answer:** the model writes an answer using only those passages, with a quote for each fact.
5. **Verify:** every quote is checked against the real document. Made-up quotes are removed.
6. **Show:** the page shows the answer, its sources, and links to each highlighted passage.

---

## Uploads

- Click the paperclip to add a file (md, txt, pdf, docx, xlsx, pptx, csv, json, html; max 5 MB).
- The file belongs to **your conversation only**. Nobody else can see it.
- It is chunked and embedded in its own temporary ChromaDB collection.
- It is **deleted** on "New conversation", after 60 minutes without use, or on restart.
- Uploads are always marked "uploaded, not official". They never override a handbook policy.
- Scanned PDFs (images of text) are refused. They need OCR first.

---

## Tests

```bash
pytest          # about 1,170 tests, no API key needed
ruff check .    # code style
```

---

## Evaluation

The dev and holdout questions were written from the handbook before the assistant was built. See [`evals/README.md`](evals/README.md) for the scoring rules.

```bash
python -m evals.retrieval_eval                  # search quality, no model calls
python -m evals.answer_eval                     # 97 dev questions, end to end
python -m evals.answer_eval --set holdout       # 26 questions kept out of tuning
python -m evals.answer_eval --set followup      # two-turn questions
python -m evals.answer_eval --set typos         # misspelled questions
python -m evals.answer_eval --injection         # prompt-injection attacks
```

**Latest results** (main model: NVIDIA Nemotron 3 Ultra):

| Set | Score |
|---|---|
| Dev | 93 / 97 |
| Holdout | 25 / 26 |
| Follow-ups | 8 / 9 |
| Typos | 10 / 11 |
| Injection attacks | 6 / 6 resisted |
| Search: right documents found | 100% |

Answers are cached in `.cache/llm/`, so re-running costs nothing.

---

## API

```bash
# ask a question
curl -X POST http://localhost:8000/api/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "How many days of annual leave do I get?"}'

# status
curl http://localhost:8000/api/health
```

Optional fields for `/api/ask`:

- `"as_of": "2027-01-15"`: answer as of another date.
- `"history": [{"question": "...", "answer": "..."}]`: for follow-up questions.

`/api/ask/stream` gives the same answer as it is being made (used by the page).

---

## Project layout

| Folder | What is inside |
|---|---|
| `rag/` | The core: reading files, document status, search, answering (plain Python) |
| `webapp/` | The Django web app: pages, API, uploads |
| `evals/` | Test questions, scoring and saved results |
| `tests/` | Automated tests |
| `data/corpus/` | The 64 handbook documents (never changed) |
| `deploy/vm/` | Auto-deploy script for the server |
| `docker/` | Start script and Caddy (HTTPS) settings |

---

## Deployment and CI/CD

The live site runs on a Google Cloud VM with Docker Compose (app, ChromaDB, and Caddy for HTTPS).

Every push to `main` goes through:

| Step | What happens |
|---|---|
| 1. Test | GitHub Actions runs lint and all tests |
| 2. Build | The Docker image is built and pushed to Google Artifact Registry |
| 3. Deploy | The VM pulls the new image within a minute, restarts and checks health. If it is not healthy, it rolls back |
| 4. Verify | GitHub waits until the live site reports the new version |

- No cloud keys are stored in GitHub. It signs in with Workload Identity Federation.
- The server's `.env` lives only on the server.
- Server setup: clone the repo, create `.env`, run `sudo sh deploy/vm/install.sh`.

---

## Known limits

- The model adds `[n]` markers in about half of its answers. Sources are always listed below the answer.
- Scanned PDFs are not supported (no OCR).
- There is no login. Anyone with the link can use the app.
- Answers depend on the model service. If it is slow, a backup model answers instead.

---

## More

- [`DECISIONS.md`](DECISIONS.md): the main design choices and why.
- [`AI_USAGE.md`](AI_USAGE.md): how AI tools were used to build this.
- [`evals/README.md`](evals/README.md): how answers are scored.
