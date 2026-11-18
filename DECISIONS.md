# Decision log

## 1. Approach in five lines

1. Every handbook file is parsed into blocks with exact character positions, then split into passages: one per section, one per table row, and one for each whole table.
2. Each document gets a state for the chosen date: in force, superseded, deprecated or not yet in force. It is worked out from the documents themselves, not trusted from the manifest.
3. Search combines keyword search (BM25) and meaning search (bge-small embeddings in ChromaDB), fused by rank. Outdated versions are pushed down, not removed.
4. The model answers only from the retrieved passages, in a fixed JSON format, quoting a sentence for each fact. Every quote is checked against the source, and fake quotes are dropped.
5. A Django page shows the answer, the verified sources, and each source with the cited passage highlighted.

## 2. Key decisions

| Decision | What else I considered | Why I chose this |
|---|---|---|
| Keep the core (`rag/`) free of Django | Logic inside the views | The core can be tested and evaluated without a web server, and the framework could be swapped. |
| Work out document states from the documents | Trust the manifest's `status` field | The manifest marks Leave Policy v1 and the 2027 parental policy as `current`. Trusting it gives wrong answers. |
| Hybrid search (BM25 + embeddings) | BM25 only, embeddings only | On the 92 dev questions that have a right document (the 5 that should be refused have none): hit@1 0.924 for hybrid, 0.870 for BM25, 0.880 for embeddings. |
| Push outdated versions down 30 places, don't remove them | Remove them, or scale scores | They are still needed for "what did v1 say?". Fused scores sit in a narrow band, so a fixed rank penalty is easier to reason about. |
| Send small documents whole, at most 2 passages per document | Plain top-k passages | Some answers need two parts of one document (city tier and hotel rate). One document was filling all 8 slots. |
| Check every quote against the source text | Trust the model's citations | The model did invent quotes (for example a per-diem of $55 that is really $65). These never reach the user. |
| Strict JSON output with a separate yes/no field | Free text | Yes/no is scored from a field, not guessed from the first word ("Not yet" is a no). |
| NVIDIA Nemotron 3 Ultra (thinking on), with a backup model | Claude Haiku via OpenRouter | Haiku scored higher (0.990) but my OpenRouter credit ran out. Nemotron is free and scored 0.959. The backup model answers when the free tier is slow. |
| Eval questions written from the corpus before building, plus a holdout set | Questions written after building, one set | Writing them first avoids testing only what the system already does. The holdout set was kept out of tuning: no change was made because of a holdout result. I did re-run it after later prompt changes, so it is not a strictly blind test. The follow-up and typo sets came later and are marked as such. |
| Typo correction with SymSpell, adding words, never replacing them | My own spell checker, or no correction | "ohw mny anual leves" missed the leave policy. SymSpell finds the typos, and an English word list stops real words from being "corrected": without it, "vacation" became "location" and "pension" became "person". |
| Uploads belong to one conversation and are deleted afterwards | One shared upload folder for everyone | Shared uploads were visible to every user and rebuilt the whole index (about a minute on a small server). Per-conversation uploads are fast: a 15 KB file is ready in 0.86 s on my laptop. |
| ChromaDB for the vectors | Keep them in memory | At 169 passages it adds nothing to speed or quality, and memory would do. I chose it because vectors survive restarts, each conversation's uploads get their own collection that is deleted in one step when the conversation ends, and the same setup still works when the handbook grows too big for memory. If ChromaDB is down, search falls back to memory. |

## 3. Things in the data or the problem that surprised me

- **The manifest is wrong for some documents.** Leave Policy v1 and the 2027 parental policy are both marked `current`. I work out states from the documents instead. This matters: without document states, the right document comes first for 0.848 of questions instead of 0.924, and an outdated version comes first for 7.6% of them.
- **The 2027 parental change depends on the birth date, not the leave date.** A baby born on 20 Dec 2026 gets 12 weeks even if leave starts in 2027. Eval question fut-05 tests this, and it passes.
- **Some questions are about outdated documents.** "How many leave days did Leave Policy v1 give?" can only be answered from v1 (12 days, eval question sup-05). So outdated documents are kept, just ranked lower.
- **Documents disagree.** A lost laptop must be reported "within 24 hours" (laptop policy) or "immediately" (incident policy). The answer gives both (conf-03 passes).
- **Some answers need a missing document.** The Holiday Calendar CSV and the Anti-Harassment Policy are referenced but not included. The answers give what the handbook does say and name the missing document.
- **"Can I deploy today?" needs a document it never names.** The runbook says "72 hours before every scheduled release", but the dates are in the Release Calendar, which search did not find. I added a rule that brings in a dated schedule for date questions. The question (date-01) now passes.
- **Documents point to each other.** Some answers sit in a document that another one only refers to. "I joined two months ago. Can I claim the home office allowance yet?" needs the Introductory Period Policy, which the home office policy points to. Without following references, 4 of the 92 dev questions miss a document they need.
- **Names are ambiguous.** There are three Mayas and three Tylers. The answer asks which one is meant (both questions pass).
- **I checked the corpus for hidden content.** I found none: no invisible PDF text, no hidden Word text, no injected instructions.

## 4. What I tried that did not work

- **A stricter yes/no rule.** I accepted a prompt rule the AI suggested: leave the yes/no field empty when an answer "depends on dates". In the next full run, the dev score went from 93 to 91: two dev questions, and one follow-up question, now had a correct answer but an empty yes/no field. I put the original rule back and fixed the real cause, which was search.
- **The first prompt-injection defence.** Telling the model that documents and questions are data was not enough: in 10 attacks on the running app, 8 were resisted, but 2 answers repeated the attacker's claim (a fake document planted in the question, and a forged earlier answer). I added an attack detector, a security note for flagged requests, and a final check that removes planted figures. I added both attacks to the injection eval, which now resists all 6 cases.
- **A query prefix for embeddings.** It helped embeddings alone (hit@1 0.880 → 0.913) but changed nothing in hybrid search, so I dropped it.
- **Uploads shared by everyone.** On the free 1 GB server, each upload rebuilt the whole index and took about a minute. I noticed this when testing the live site, and moved to per-conversation uploads.
- **Plain substring matching in the grader.** "2" matched "$2,500" and "2026". I changed it to whole-token matching.

## 5. Results

Model: `nvidia/nemotron-3-ultra-550b-a55b`, thinking on, through NVIDIA's free API (`integrate.api.nvidia.com`). A full dev run makes **97 model calls**, one per question. The search eval makes none.

| Metric | Dev (97) | Holdout (26) |
|---|---|---|
| Accuracy | **0.959** (93/97) | **0.962** (25/26) |
| Answer / refuse / clarify | 0.956 / 1.000 / 1.000 | 0.957 / 1.000 / 1.000 |
| Yes/no correct | 0.923 | 1.000 |
| Answers with a supporting citation | 1.000 | 1.000 |
| Citation precision (strict) | 0.912 | 0.979 |
| Quotes dropped as fake | 1 | 0 |
| Search: needed documents found | 1.000 | 1.000 |

Search alone, on the 92 dev questions that have a right document (no model calls):

| Setup | hit@1 | recall@5 | MRR | Context recall |
|---|---|---|---|---|
| BM25 only (keywords) | 0.870 | 0.947 | 0.922 | 0.991 |
| Embeddings only, MiniLM | 0.880 | 0.957 | 0.931 | 0.973 |
| Embeddings only, bge-small | 0.880 | 0.946 | 0.929 | 0.989 |
| Embeddings only, bge-small + query prefix | 0.913 | 0.957 | 0.945 | 0.989 |
| Hybrid, MiniLM | 0.891 | 0.973 | 0.942 | 0.989 |
| **Hybrid, bge-small (chosen)** | **0.924** | 0.967 | **0.960** | **1.000** |
| Hybrid, bge-small + query prefix | 0.924 | 0.967 | 0.960 | 1.000 |

- **hit@1:** the right document is the first result.
- **recall@5:** the right documents are in the top 5.
- **MRR:** how high the first right document ranks, on average (1.0 = always first).
- **Context recall:** the documents the answer needs reach the model.

Why hybrid BM25 + `BAAI/bge-small-en-v1.5`, with no prefix:

- **Hybrid beats either method alone** on hit@1, MRR and context recall, for both embedding models.
- **bge-small beats MiniLM** in hybrid on 3 of 4 measures. MiniLM is only slightly ahead on recall@5 (0.973 vs 0.967), and bge-small is the only setup where every needed document reaches the model (1.000).
- **No query prefix:** the prefix helps embeddings alone (hit@1 0.880 → 0.913), but in hybrid it scores exactly the same, so I left it out to keep things simple.

## 6. What I would do with one more week

1. **Add a re-ranker after search.** Hit@1 is 0.924, so for 7 of the 92 dev questions the first result is the wrong document. A cross-encoder re-ranker (for example `bge-reranker`) would re-score the top 20 passages by reading each one together with the question, which targets exactly these misses. I would keep it only if the search eval shows a gain.
2. **Compare other LLMs on the same eval.** Model choice clearly matters: Claude Haiku scored 0.990 against Nemotron's 0.959. I would run OpenAI's GPT models, Google's Gemini models and Claude models through the same 97 dev questions (the eval already takes `--models`), and pick on accuracy, citation precision, speed and cost per question.
3. **Rewrite questions with an LLM before searching.** Today typos are fixed by a spell checker and follow-ups by adding the previous question to the search. A short LLM step could turn "and who approves it?" into a full question and expand "PTO" into "paid time off". I would measure it on the follow-up set (8/9 now) and the typo set (10/11 now).
4. **Remember the user across conversations (for example with mem0).** The app keeps only the last 3 turns of one conversation. Many answers depend on who is asking: the per-diem depends on grade and city, and remote work on the team. A memory layer could remember "I'm an L2 in Denver" so the answer fits without asking again.
5. **Read scanned PDFs with OCR.** Scanned PDFs (images of text) are refused today. An OCR step (for example Tesseract) would turn them into text before chunking, so scanned policies could be uploaded too.
6. **Keep the handbook in object storage (S3 or Google Cloud Storage).** The handbook is copied into the Docker image, so changing one policy means building and deploying a new image. With the files in a storage bucket, the app could re-index when a file changes, and old versions would be kept by the bucket.
7. **Add monitoring and alerts.** Today I only notice problems by testing by hand. I would track errors, answer time, backup-model use and refusals, and get an alert when the site is down or the error rate rises.

## 7. Time spent

Rough estimates (the work ran from 17 to 18 Nov):

| Part | Hours |
|---|---|
| Reading the brief and corpus, writing eval questions | 1.5 |
| Architecture design | 3 |
| Parsing, document states, chunking | 2 |
| Search and search eval | 1.5 |
| Answer step, model provider, answer eval | 2 |
| Web app | 1.5 |
| Fixes from testing (follow-ups, typos, uploads, injection) | 1.5 |
| Deployment and CI/CD | 2 |
| Write-up | 0.5 |
| **Total** | **about 15.5** |
