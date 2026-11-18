# AI usage

## Tools I used

- **Claude Code** (Anthropic), inside VS Code.
- **A separate AI chat session as a reviewer.** I pasted its reviews of my work back in and checked each point before acting on it.
- **NVIDIA's model API** (Nemotron) for the app's LLM calls.

## What I used them for

| Task | Tool | How much of the result I kept |
|---|---|---|
| Reading the brief and auditing all 64 corpus files for traps and hidden content | Claude Code | All of the findings. Each trap it found became an eval question, so the checks are in the test set |
| Writing the eval questions | Claude Code | All of them. Tests make sure every trap category, every document and every file format has a question |
| Parsers, document states, chunking, search | Claude Code | Most of the code, after tests and the search eval showed it worked |
| Answer step, prompt, quote checking | Claude Code | Most of the code. I reverted one prompt change that made results worse (see below) |
| Web app (Django, page, uploads) | Claude Code | Most of it. Many fixes came from my own testing in the browser |
| Deployment and CI/CD | Me, with Claude Code | I chose the hosting, the VM size and a pull-based deploy, and did all account, key and permission steps. Claude Code wrote the pipeline and deploy scripts and I ran the cloud commands. |
| Reviewing the work | Second AI session | Only the points that held up when checked. Some were already done, some I rejected |
| README, DECISIONS.md, this file | I provided and highlighted the points, formatted and improved by Claude Code | Most of the text. I checked every number against the saved results |

## One thing the AI got wrong

**A prompt change that made the yes/no answers worse.**

- **What it suggested:** the "Can I deploy today?" question was answered "Yes" during a deploy freeze. The AI added a rule: leave the yes/no field empty when an answer "depends on dates".
- **How it showed up:** the next full eval run went down from 93/97 to 91/97. Two dev questions and one follow-up question now had a correct answer text but an empty yes/no field, for example "Can I claim the home office allowance a second time?". The rule was too broad: the model treated every condition as a date condition.
- **What I did instead:** I had the original yes/no rule put back. The real cause of the freeze bug was search: the Release Calendar was never found. Fixing that fixed the question, without a risky prompt rule.

**Other mistakes I caught:**

- **The grader used plain substring matching,** so "2" matched "$2,500". Changed to whole-token matching.
- **Its first upload design shared files between all users** and rebuilt the whole index, which took a minute on a small server. I asked for uploads per conversation instead.
- **The auto-deploy script picked the image by list position.** Compose lists two images in no fixed order, so it sometimes "redeployed" for no reason. This showed up in the server's deploy log after the first pipeline run.
- **Its first prompt-injection defence still leaked the attacker's numbers.** In 10 attacks on the running app, 2 answers repeated the attacker's claim. I added an attack detector, a security note for flagged requests, and a final check that removes any sentence repeating a planted figure. Both attacks went into the injection eval, which now resists all 6 cases.
- **Its tests passed even when the code was broken on purpose.** I had small bugs planted in the code (a wrong PDF page number, wrong amendment dates, unused chunking code) to see if the tests would notice. Some did not, so "all tests pass" proved less than it looked. Those tests were made stricter until each planted bug made them fail.

## What I did without AI

### Architecture, development, testing and deployment

- **Architecture Decisions:** Selected Django as the application framework, NVIDIA as the model provider, ChromaDB as the vector store, and a single VM running Docker Compose for deployment. Designed uploads to be conversation-specific and automatically deleted afterwards.

- **RAG Architecture:** Designed the document-processing and retrieval pipeline around normalized document representations, structure-aware chunking, metadata, hybrid retrieval, and evidence-based responses.

- **Chunking Strategy:** Evaluated fixed-size chunking with overlap versus structure-aware chunking and chose the latter to preserve document context.

- **Retrieval Improvements:** Reviewed the query-processing pipeline and prioritized embeddings.

- **Evaluation:** Required a full evaluation before every push, an eval run before and after each prompt change, and prompt-injection attacks against the running app. Search is measured with hit@1, recall@5, MRR and context recall; answers with accuracy, yes/no verdicts and citation precision, on a dev set and a separate holdout set.

- **Spell Checking:** Added spell-checking functionality to improve query handling and user experience.

- **Manual Testing and Bug Fixes:** Tested the application hands-on, identified issues, and reported them for resolution. This led to fixes for:
  - Chat history disappearing after opening a source and returning.
  - Repeated questions not being served from the cache.
  - Simple greetings incorrectly triggering source-verification warnings.
  - Markdown formatting displaying raw asterisks instead of bold text.
  - Misspelled questions failing to retrieve relevant answers.
  - The fallback model exposing raw JSON in its responses.
  - Source cards displaying entire documents instead of relevant excerpts.
  - Outdated policy information taking precedence over newer rules in future-dated questions.
  - Slow document uploads on the deployed server.

- **DevOps and Deployment:** Independently handled the DevOps work, server setup, Docker Compose configuration, and deployment of the complete application.
