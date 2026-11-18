import json
import logging
import mimetypes
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlencode

from django.conf import settings
from django.http import (
    FileResponse,
    Http404,
    HttpResponseNotAllowed,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from rag.answer import MAX_QUESTION_CHARS, SMALL_TALK, STATE_LABEL, answer_question, clean_history, is_follow_up
from rag.uploads import ALLOWED_EXTENSIONS, UploadError, valid_session_id
from rag.validity import describe
from webapp.handbook import services
from webapp.handbook.rendering import render_blocks

log = logging.getLogger(__name__)

EXAMPLES = [
    "How many days of annual leave do I get?",
    "Can I work from Portugal for a month?",
    "What is the per-diem for an L1 employee visiting the Denver office?",
    "How much leave does a secondary caregiver get?",
    "Does a $4,000 trip booked through VoyageHub need VP approval?",
    "Who is Maya?",
]


# ---------- rate limiting (protects the model provider's quota) ----------

_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _client_ip(request) -> str:
    """The visitor's address. Behind a trusted proxy (TRUST_PROXY=1) it is the last X-Forwarded-For
    entry, the one the proxy added itself; earlier entries come from the client and can be faked."""
    if getattr(settings, "TRUST_PROXY", False):
        forwarded = [p.strip() for p in request.META.get("HTTP_X_FORWARDED_FOR", "").split(",") if p.strip()]
        if forwarded:
            return forwarded[-1]
    return request.META.get("REMOTE_ADDR", "unknown")


def _rate_limited(request, bucket: str = "ask") -> bool:
    """A simple per-address sliding window: at most ASK_RATE_LIMIT_PER_MIN questions (or
    UPLOAD_RATE_LIMIT_PER_MIN uploads) a minute."""
    limit = getattr(settings, "UPLOAD_RATE_LIMIT_PER_MIN" if bucket == "upload" else "ASK_RATE_LIMIT_PER_MIN", 20)
    if not limit:
        return False
    key = f"{bucket}:{_client_ip(request)}"
    now = time.monotonic()
    with _hits_lock:
        window = _hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= limit:
            return True
        window.append(now)
        return False


def _error(message: str, status: int) -> JsonResponse:
    return JsonResponse({"error": message}, status=status)


def _conversation_id(request, body=None) -> str | None:
    """The page's conversation id: header X-Conversation-Id, a "conversation" field, or ?conversation=."""
    value = request.headers.get("X-Conversation-Id") or (body or {}).get("conversation") or request.GET.get("conversation")
    return valid_session_id(value)


def _source_url_for(conversation: str | None, uploaded: set[str]):
    """Links that open a citation; a document uploaded in this conversation carries the conversation id."""
    def source_url(doc_id: str, start: int | None = None, end: int | None = None) -> str:
        params = {}
        if start is not None and end is not None:
            params |= {"start": start, "end": end}
        if doc_id in uploaded and conversation:
            params["conversation"] = conversation
        url = reverse("handbook:source", args=[doc_id])
        return f"{url}?{urlencode(params)}" + ("#hl" if "start" in params else "") if params else url
    return source_url


def _parse_request(request):
    """(question, as_of date, history, conversation id) from a JSON or form POST, or an error response."""
    def fail(response):
        return None, None, None, None, response

    if request.method != "POST":
        return fail(HttpResponseNotAllowed(["POST"]))
    if request.content_type == "application/json":
        try:
            body = json.loads(request.body or b"{}")
        except json.JSONDecodeError:
            return fail(_error("The request body is not valid JSON.", 400))
        if not isinstance(body, dict):
            return fail(_error("Send a JSON object like {\"question\": \"...\"}.", 400))
    else:
        body = request.POST
    question = str(body.get("question") or "").strip()
    if not question:
        return fail(_error("Please type a question.", 400))
    if len(question) > MAX_QUESTION_CHARS:
        return fail(_error(f"Please keep questions under {MAX_QUESTION_CHARS} characters.", 400))
    try:
        as_of = services.parse_as_of(body.get("as_of"))
    except ValueError:
        return fail(_error("The date must look like 2027-01-15.", 400))
    history = body.get("history") if hasattr(body, "get") else None
    if history is not None and not isinstance(history, list):
        return fail(_error("history must be a list of {\"question\", \"answer\"} turns.", 400))
    if _rate_limited(request):
        return fail(_error("Too many questions in a short time. Please wait a minute and try again.", 429))
    if services.get_client() is None:
        return fail(_error("The assistant is not configured yet: " + (services.client_error() or ""), 503))
    return question, as_of, clean_history(history), _conversation_id(request, body), None


# ---------- pages ----------

@require_GET
@ensure_csrf_cookie
def home(request):
    return render(request, "handbook/ask.html", {
        "examples": EXAMPLES, "max_chars": MAX_QUESTION_CHARS,
        "upload_accept": ",".join(ALLOWED_EXTENSIONS), "upload_max_mb": settings.UPLOAD_MAX_BYTES // 1_000_000,
        "upload_ttl_minutes": settings.UPLOAD_SESSION_TTL_MINUTES,
    })


def _find_document(request, doc_id: str):
    """(document, folder holding its original file): a handbook document, or one uploaded in this conversation."""
    index = services.get_index()
    if doc_id in index.docs:
        return index.docs[doc_id], settings.CORPUS_DIR
    conversation = services.uploads().get(_conversation_id(request))
    if conversation and doc_id in conversation.docs:
        return conversation.docs[doc_id], conversation.folder
    raise Http404("No such document in the handbook (an upload is only visible in its own conversation).")


@require_GET
def source(request, doc_id: str):
    index = services.get_index()
    doc, folder = _find_document(request, doc_id)
    try:
        start = int(request.GET["start"]) if "start" in request.GET else None
        end = int(request.GET["end"]) if "end" in request.GET else None
    except ValueError:
        start = end = None
    if start is None or end is None or not 0 <= start < end <= len(doc.text):
        start = end = None
    status = None if doc.meta.uploaded else index.statuses().get(doc_id)
    titles = {d: x.meta.title for d, x in index.docs.items()}
    conversation = _conversation_id(request)
    file_url = reverse("handbook:source_file", args=[doc_id])
    if doc.meta.uploaded and conversation:
        file_url += "?" + urlencode({"conversation": conversation})
    return render(request, "handbook/source.html", {
        "doc": doc,
        "meta": doc.meta,
        "state": status.state.value if status else "in_force",
        "state_label": STATE_LABEL[status.state] if status else "in force",
        "status_note": describe(status, titles) if status else "",
        # say something only when it matters: outdated, upcoming, amended, or informal notes
        "show_notice": bool(status) and describe(status, titles) != "In force.",
        "blocks": render_blocks(doc, start, end),
        "highlighted": start is not None,
        "has_file": (folder / doc.meta.path).is_file(),
        "file_url": file_url,
        "uploaded": doc.meta.uploaded,
        "reference_date": index.date_for(),
    })


@require_GET
def source_file(request, doc_id: str):
    """The original file (PDF, Word, ...) so a reader can check the source itself."""
    doc, folder = _find_document(request, doc_id)
    root = Path(folder).resolve()
    path = (root / doc.meta.path).resolve()
    if root not in path.parents or not path.is_file():  # only files inside the corpus (or uploads), never a user path
        raise Http404("The original file is not available.")
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    inline = content_type in ("application/pdf", "text/plain", "text/markdown", "text/csv")
    return FileResponse(path.open("rb"), content_type=content_type, as_attachment=not inline, filename=path.name)


# ---------- API ----------
# No login or session state exists, so these read-only endpoints are exempt from CSRF; that lets
# scripts and reviewers call them with curl. Anything that changes state (uploads) keeps CSRF.

@csrf_exempt
def api_ask(request):
    """POST {"question": "...", "as_of": "2027-01-15"?, "history": [...]?, "conversation": "<id>"?} -> the answer.

    `history` (up to 3 earlier turns) lets follow-ups like "which one lives in Austin?" work; the
    conversation id (or header X-Conversation-Id) adds the documents uploaded in that conversation.
    """
    question, as_of, history, conversation, error = _parse_request(request)
    if error:
        return error
    uploads = services.uploads().get(conversation)
    answer = answer_question(question, services.get_index(), services.get_client(), as_of, history=history,
                             uploads=uploads)
    status = 502 if answer.answerable is None and answer.error else 200
    source_url = _source_url_for(conversation, set(uploads.docs) if uploads else set())
    return JsonResponse(services.answer_json(answer, source_url), status=status)


@csrf_exempt
def api_ask_stream(request):
    """Same as api_ask, streamed as newline-delimited JSON so the page can show progress:

    {"type": "retrieval", ...}   about 0.1 s: which documents are being read
    {"type": "answer", ...}      when the model has answered and the quotes are checked
    """
    question, as_of, history, conversation, error = _parse_request(request)
    if error:
        return error
    index, client = services.get_index(), services.get_client()
    uploads = services.uploads().get(conversation)
    uploaded = set(uploads.docs) if uploads else set()
    source_url = _source_url_for(conversation, uploaded)
    if not is_follow_up(question):
        history = []  # a complete question is answered on its own (same prompt every time, so cacheable)
    previous = history[-1]["question"] if history else None

    def events():
        started = time.perf_counter()
        if SMALL_TALK.match(question):  # greetings: no search, no model call
            answer = answer_question(question, index, client, as_of)
            yield json.dumps({"type": "retrieval", "reference_date": index.date_for(as_of).isoformat(), "context": []}) + "\n"
            yield json.dumps({"type": "answer", **services.answer_json(answer, source_url)}) + "\n"
            return
        try:
            retrieval = index.retrieve(question, as_of, follow_up_of=previous, uploads=uploads)
            yield json.dumps({"type": "retrieval", "reference_date": retrieval.reference_date.isoformat(),
                              "context": services.context_json(retrieval.context, uploaded)}) + "\n"
            answer = answer_question(question, index, client, as_of, retrieval=retrieval, history=history)
            payload = services.answer_json(answer, source_url)
            log.info("answered in %.1fs (cached=%s, citations=%d, error=%s)", time.perf_counter() - started,
                     answer.cached, len(answer.citations), bool(answer.error))
            yield json.dumps({"type": "answer", **payload}) + "\n"
        except Exception:  # headers are already sent, so report the failure inside the stream
            log.exception("streamed answer failed")
            yield json.dumps({"type": "error", "error": "Something went wrong while answering. Please try again."}) + "\n"

    response = StreamingHttpResponse(events(), content_type="application/x-ndjson")
    response["Cache-Control"] = "no-store"
    response["X-Accel-Buffering"] = "no"  # stop proxies from holding the first line back
    return response


@require_GET
def api_health(request):
    """Status, and a warm-up: the page calls this on load so the index is ready by the first question."""
    index = services.get_index()
    client = services.get_client()
    return JsonResponse({
        "status": "ok" if client else "degraded",
        "documents": len(index.docs),
        "passages": len(index.passages),
        "search_mode": index.mode,
        "vector_store": index.retriever.vector_store_name,
        "model": getattr(client, "model", None),
        "fallback_model": getattr(getattr(client, "fallback", None), "model", None),
        "model_error": services.client_error(),
        "today": index.date_for().isoformat(),
        "version": os.environ.get("APP_VERSION", "dev"),
    })


# ---------- uploads: one conversation's own documents (change state, so CSRF protection stays on) ----------

def _upload_json(doc, conversation: str, created: bool | None = None, passages: int | None = None) -> dict:
    out = {"doc_id": doc.doc_id, "title": doc.meta.title, "format": doc.meta.format,
           "url": _source_url_for(conversation, {doc.doc_id})(doc.doc_id)}
    if created is not None:
        out |= {"created": created, "passages": passages}
    return out


@require_POST
def api_upload(request):
    """POST multipart/form-data with `file` (and the conversation id) -> the document, searchable at once
    in this conversation only."""
    max_bytes = settings.UPLOAD_MAX_BYTES
    try:
        declared = int(request.META.get("CONTENT_LENGTH") or 0)
    except ValueError:
        declared = 0
    if declared > max_bytes + 100_000:  # refuse before Django reads a huge body to disk
        return _error(f"The file is larger than {max_bytes // 1_000_000} MB.", 413)
    if _rate_limited(request, "upload"):
        return _error("Too many uploads in a short time. Please wait a minute and try again.", 429)
    conversation = _conversation_id(request, request.POST)
    if not conversation:
        return _error("Missing conversation id; reload the page and try again.", 400)
    upload = request.FILES.get("file")
    if upload is None:
        return _error("Choose a file to upload.", 400)
    if upload.size > max_bytes:
        return _error(f"The file is larger than {max_bytes // 1_000_000} MB.", 413)
    store = services.uploads()
    try:
        doc, created = store.add(conversation, upload.name, upload.read(max_bytes + 1), services.get_index().date_for())
    except UploadError as exc:
        return _error(str(exc), 400)
    passages = len(store.get(conversation).passages_by_doc.get(doc.doc_id, []))
    log.info("upload %s: %s (%s, %d passages)", "added" if created else "already present", doc.doc_id,
             doc.meta.format, passages)
    return JsonResponse(_upload_json(doc, conversation, created, passages), status=201 if created else 200)


@require_GET
def api_uploads(request):
    """The documents uploaded in this conversation."""
    conversation = _conversation_id(request)
    docs = services.uploads().documents(conversation)
    return JsonResponse({"documents": [_upload_json(d, conversation) for d in docs],
                         "accept": ALLOWED_EXTENSIONS, "max_mb": settings.UPLOAD_MAX_BYTES // 1_000_000,
                         "keep_minutes": settings.UPLOAD_SESSION_TTL_MINUTES})


@require_POST
def api_upload_delete(request, doc_id: str):
    if not services.uploads().remove(_conversation_id(request, request.POST), doc_id):
        return _error("No such uploaded document in this conversation.", 404)
    return JsonResponse({"removed": doc_id})


@require_POST
def api_conversation_clear(request):
    """Delete everything uploaded in this conversation (files, chunks, vectors): "New conversation"."""
    conversation = _conversation_id(request, request.POST)
    if not conversation:
        return _error("Missing conversation id.", 400)
    services.uploads().clear(conversation)
    return JsonResponse({"cleared": True})
