"""The web app: pages, the JSON and streaming APIs, the source viewer, errors and limits.

Uses Django's test client, the real handbook (keyword search, no model download) and a fake
language model, so it costs nothing and needs no network.
"""

import json
from datetime import date

import pytest
from django.test import Client, override_settings

from rag.index import HandbookIndex
from rag.llm import Completion, LLMError
from webapp.handbook import services, views

V2 = "Full-time employees accrue 16 days of annual leave per calendar year"


class FakeClient:
    model = "fake/model"

    def __init__(self, content=None, error=None):
        self.content = content or {
            "answer": "Full-time employees get 16 days of annual leave a year.", "verdict": None,
            "answerable": True, "needs_clarification": False,
            "citations": [{"doc_id": "hr-leave-policy-v2", "quote": V2}]}
        self.error = error
        self.calls = 0

    def complete(self, messages, schema, max_tokens=4000):
        self.calls += 1
        if self.error:
            raise self.error
        return Completion(self.content, self.model, "fake", 100, 20, 0.0, 3.0, False)


@pytest.fixture(scope="module")
def bm25_index(docs):
    return HandbookIndex.build(list(docs.values()), date(2026, 11, 17), mode="bm25")


@pytest.fixture
def app(bm25_index, monkeypatch, tmp_path):
    """Point the web app at the test index, a fake model and an empty upload folder; reset the rate limiter."""
    from django.conf import settings
    fake = FakeClient()
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(services, "_index_uploads", 0)
    monkeypatch.setattr(services, "_index", bm25_index)
    monkeypatch.setattr(services, "_client", fake)
    monkeypatch.setattr(services, "_client_error", None)
    views._hits.clear()
    return fake


@pytest.fixture
def web():
    return Client()


def ask(web, question="How many days of annual leave do I get?", stream=False, **extra):
    url = "/api/ask/stream" if stream else "/api/ask"
    return web.post(url, data=json.dumps({"question": question, **extra}), content_type="application/json")


# ---------- pages ----------

def test_home_page(web, app):
    r = web.get("/")
    assert r.status_code == 200
    html = r.content.decode()
    assert "Ask the employee handbook" in html and 'id="ask-form"' in html
    assert "/api/ask/stream" in html and "app.js" in html
    assert "csrftoken" in r.cookies


def test_security_headers(web, app):
    r = web.get("/")
    assert r["X-Frame-Options"] == "DENY" and r["X-Content-Type-Options"] == "nosniff"


def test_health_warms_up_and_reports(web, app):
    body = web.get("/api/health").json()
    assert body["status"] == "ok" and body["documents"] == 64 and body["model"] == "fake/model"


# ---------- the ask API ----------

def test_ask_returns_answer_with_verified_citation_links(web, app):
    r = ask(web)
    assert r.status_code == 200
    body = r.json()
    assert body["answerable"] is True and "16 days" in body["answer"]
    (cite,) = body["citations"]
    assert cite["doc_id"] == "hr-leave-policy-v2" and cite["state"] == "in_force" and cite["n"] == 1
    assert cite["url"].startswith("/source/hr-leave-policy-v2/?start=") and cite["url"].endswith("#hl")
    assert body["context"] and body["reference_date"]


def test_stream_sends_retrieval_first_then_the_answer(web, app):
    r = ask(web, stream=True)
    assert r.status_code == 200 and r["Content-Type"] == "application/x-ndjson"
    events = [json.loads(line) for line in b"".join(r.streaming_content).decode().splitlines() if line]
    assert [e["type"] for e in events] == ["retrieval", "answer"]
    assert any(c["doc_id"] == "hr-leave-policy-v2" for c in events[0]["context"])
    assert events[1]["citations"][0]["doc_id"] == "hr-leave-policy-v2"
    assert app.calls == 1  # search ran once, shared by both events


def test_as_of_date_changes_the_states(web, app):
    body = ask(web, "How much leave does a secondary caregiver get?", as_of="2027-02-01").json()
    states = {c["doc_id"]: c["state"] for c in body["context"]}
    assert body["reference_date"] == "2027-02-01"
    assert states.get("hr-parental-leave-policy-2027") == "in_force"


@pytest.mark.parametrize("payload,status,message", [
    ({"question": "   "}, 400, "type a question"),
    ({"question": "x" * 2001}, 400, "under 2000"),
    ({"question": "leave", "as_of": "15/01/2027"}, 400, "2027-01-15"),
    ({"question": "leave", "as_of": "1850-01-01"}, 400, "2027-01-15"),
])
def test_bad_input_is_rejected_clearly(web, app, payload, status, message):
    r = web.post("/api/ask", data=json.dumps(payload), content_type="application/json")
    assert r.status_code == status and message in r.json()["error"]
    assert app.calls == 0


def test_bad_json_and_wrong_method(web, app):
    assert web.post("/api/ask", data="not json", content_type="application/json").status_code == 400
    assert web.post("/api/ask", data="[1]", content_type="application/json").status_code == 400
    assert web.get("/api/ask").status_code == 405


def test_form_posts_work_too(web, app):
    r = web.post("/api/ask", {"question": "How many days of annual leave do I get?"})
    assert r.status_code == 200 and r.json()["answerable"] is True


def test_model_failure_is_a_clean_502(web, app, monkeypatch):
    monkeypatch.setattr(services, "_client", FakeClient(error=LLMError("model call failed: HTTP 503")))
    r = ask(web)
    assert r.status_code == 502
    body = r.json()
    assert body["answerable"] is None and "could not answer" in body["answer"]


def test_missing_model_settings_is_a_clear_503(web, app, monkeypatch):
    monkeypatch.setattr(services, "_client", None)
    monkeypatch.setattr(services, "_client_error", "missing setting(s): LLM_API_KEY")
    r = ask(web)
    assert r.status_code == 503 and "LLM_API_KEY" in r.json()["error"]


@override_settings(ASK_RATE_LIMIT_PER_MIN=3)
def test_rate_limit(web, app):
    codes = [ask(web).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    assert "wait a minute" in ask(web).json()["error"]


def test_answer_text_is_never_turned_into_html(web, app, monkeypatch):
    hostile = FakeClient({"answer": "<img src=x onerror=alert(1)>", "verdict": None, "answerable": True,
                          "needs_clarification": False, "citations": []})
    monkeypatch.setattr(services, "_client", hostile)
    body = ask(web).json()
    assert body["answer"] == "<img src=x onerror=alert(1)>"  # returned as data; the page sets textContent
    js = open("webapp/handbook/static/handbook/app.js").read()
    assert "innerHTML" not in js.replace("never innerHTML", "")


# ---------- source viewer ----------

def test_source_page_highlights_the_cited_passage(web, app, docs):
    text = docs["hr-leave-policy-v2"].text
    start = text.index(V2)
    r = web.get(f"/source/hr-leave-policy-v2/?start={start}&end={start + len(V2)}")
    html = r.content.decode()
    assert r.status_code == 200
    assert f'<mark class="hl">{V2}</mark>' in html
    assert "Leave Policy v2" in html and "in force" in html


def test_source_page_explains_outdated_documents(web, app):
    html = web.get("/source/hr-leave-policy-v1/").content.decode()
    assert "Superseded by Leave Policy v2." in html and 'class="notice notice-superseded"' in html


def test_source_page_renders_tables(web, app):
    html = web.get("/source/hr-travel-policy/").content.decode()
    assert "<th scope=\"col\">City tier</th>" in html and "<td>$275</td>" in html


@pytest.mark.parametrize("query", ["?start=abc&end=9", "?start=50&end=10", "?start=0&end=999999", ""])
def test_source_page_ignores_bad_offsets(web, app, query):
    r = web.get(f"/source/hr-leave-policy-v2/{query}")
    assert r.status_code == 200 and "<mark" not in r.content.decode()


def test_original_file_download(web, app):
    r = web.get("/source/hr-leave-policy-v2/file")
    assert r.status_code == 200 and r["Content-Type"] == "application/pdf"
    assert b"".join(r.streaming_content).startswith(b"%PDF")
    r = web.get("/source/hr-travel-policy/file")
    assert "attachment" in r["Content-Disposition"]  # Word files download rather than display


@pytest.mark.parametrize("path", ["/source/no-such-doc/", "/source/no-such-doc/file", "/source/../settings/"])
def test_unknown_documents_and_paths_are_404(web, app, path):
    assert web.get(path).status_code == 404


def test_document_text_is_escaped_on_the_source_page(web, app, docs, monkeypatch):
    from dataclasses import replace

    from rag.loaders import RawBlock, build_text
    from rag.models import Document
    text, blocks = build_text([RawBlock("paragraph", "<script>alert('x')</script> Visitors sign in.")], title="Evil")
    evil = Document(replace(docs["hr-lab-safety"].meta, doc_id="evil-doc", title="Evil"), text, blocks)
    index = HandbookIndex.build(list(docs.values()) + [evil], date(2026, 11, 17), mode="bm25")
    monkeypatch.setattr(services, "_index", index)
    html = web.get("/source/evil-doc/").content.decode()
    assert "<script>alert" not in html and "&lt;script&gt;" in html


def test_unexpected_failure_mid_stream_ends_with_an_error_event(web, app, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(views, "answer_question", broken)
    r = ask(web, stream=True)
    events = [json.loads(line) for line in b"".join(r.streaming_content).decode().splitlines() if line]
    assert [e["type"] for e in events] == ["retrieval", "error"]
    assert "try again" in events[1]["error"] and "disk" not in events[1]["error"]  # no internals leaked


def test_original_file_never_escapes_the_corpus_folder(web, app, docs, monkeypatch):
    """A document whose metadata points outside the corpus (e.g. a hostile upload) must not serve that file."""
    from dataclasses import replace

    from rag.models import Document
    sneaky = Document(replace(docs["hr-lab-safety"].meta, doc_id="sneaky-doc", path="../../.env"),
                      docs["hr-lab-safety"].text, docs["hr-lab-safety"].blocks)
    index = HandbookIndex.build(list(docs.values()) + [sneaky], date(2026, 11, 17), mode="bm25")
    monkeypatch.setattr(services, "_index", index)
    assert web.get("/source/sneaky-doc/file").status_code == 404


def test_page_has_conversation_controls_and_no_native_required_bubble(web, app):
    html = web.get("/").content.decode()
    assert 'id="new-chat"' in html
    assert "required" not in html.split('id="question"')[1].split(">")[0]  # no browser "fill in this field" bubble
    js = open("webapp/handbook/static/handbook/app.js").read()
    assert "localStorage" in js and "isYesNoQuestion" in js


def test_follow_up_history_reaches_the_model(web, app, monkeypatch):
    sent = []

    class Recording(FakeClient):
        def complete(self, messages, schema, max_tokens=4000):
            sent.append(messages[1]["content"])
            return super().complete(messages, schema, max_tokens)

    monkeypatch.setattr(services, "_client", Recording())
    history = [{"question": "Who is Maya?", "answer": "There are three: Rivera, Brooks and Park."}]
    r = ask(web, "which one lives in austin", stream=True, history=history)
    events = [json.loads(line) for line in b"".join(r.streaming_content).decode().splitlines() if line]
    assert "org-chart" in {c["doc_id"] for c in events[0]["context"]}  # search used the previous question
    assert "Question: Who is Maya?" in sent[0]  # and the model saw the conversation


def test_bad_history_is_rejected(web, app):
    r = web.post("/api/ask", data=json.dumps({"question": "x", "history": "Who is Maya?"}),
                 content_type="application/json")
    assert r.status_code == 400 and "history" in r.json()["error"]


def test_answers_render_bold_safely_and_drop_pasted_citations():
    js = open("webapp/handbook/static/handbook/app.js").read()
    assert "function renderInline" in js and 'el("strong"' in js  # **bold** becomes <strong>, built as elements
    assert "function cleanAnswer" in js and "CITATION_TAIL" in js
    assert "innerHTML" not in js.replace("never innerHTML", "")


def test_static_files_are_fingerprinted_so_browsers_never_keep_an_old_version(web, app, tmp_path):
    import re
    html = web.get("/").content.decode()
    css = re.search(r'app\.css\?v=([0-9a-f]{10})', html)
    js = re.search(r'app\.js\?v=([0-9a-f]{10})', html)
    assert css and js, "static URLs must carry a content fingerprint"
    from webapp.handbook.templatetags.handbook_static import static_v
    assert static_v("handbook/app.js").endswith(js.group(1))
    assert static_v("handbook/no-such-file.js") == "/static/handbook/no-such-file.js"


def test_small_talk_streams_a_help_reply_without_a_model_call(web, app):
    r = ask(web, "hi", stream=True)
    events = [json.loads(line) for line in b"".join(r.streaming_content).decode().splitlines() if line]
    assert [e["type"] for e in events] == ["retrieval", "answer"]
    assert events[0]["context"] == [] and events[1]["small_talk"] is True
    assert events[1]["unsupported"] is False and app.calls == 0


def test_complete_question_drops_history_before_the_model(web, app, monkeypatch):
    sent = []

    class Recording(FakeClient):
        def complete(self, messages, schema, max_tokens=4000):
            sent.append(messages[1]["content"])
            return super().complete(messages, schema, max_tokens)

    monkeypatch.setattr(services, "_client", Recording())
    history = [{"question": "Who is Maya?", "answer": "There are three: Rivera, Brooks and Park."}]
    b"".join(ask(web, stream=True).streaming_content)
    b"".join(ask(web, stream=True, history=history).streaming_content)
    assert len(sent) == 2 and sent[0] == sent[1] and "<conversation>" not in sent[1]


# ---------- uploads ----------

NOTE = b"# Team Offsite\n\nThe platform team offsite is in Lisbon on 2027-03-10. Flights are booked by People Ops.\n"


def upload(web, name="offsite.md", data=NOTE):
    from django.core.files.uploadedfile import SimpleUploadedFile
    return web.post("/api/upload", {"file": SimpleUploadedFile(name, data)})


def test_uploaded_document_can_be_asked_about_straight_away(web, app):
    r = upload(web)
    assert r.status_code == 201, r.content
    doc_id = r.json()["doc_id"]
    assert r.json()["created"] and r.json()["passages"] >= 1
    events = [json.loads(line) for line in
              b"".join(ask(web, "Where is the platform team offsite?", stream=True).streaming_content)
              .decode().splitlines() if line]
    context = {c["doc_id"]: c for c in events[0]["context"]}
    assert doc_id in context and context[doc_id]["uploaded"] is True
    assert [d["doc_id"] for d in web.get("/api/uploads").json()["documents"]] == [doc_id]


def test_uploaded_source_page_says_it_is_not_official_and_serves_the_file(web, app):
    doc_id = upload(web).json()["doc_id"]
    page = web.get(f"/source/{doc_id}/").content.decode()
    assert "not part of the official handbook" in page and "Lisbon" in page
    original = web.get(f"/source/{doc_id}/file")
    assert original.status_code == 200 and b"".join(original.streaming_content) == NOTE


def test_upload_needs_a_csrf_token(app):
    from django.core.files.uploadedfile import SimpleUploadedFile
    strict = Client(enforce_csrf_checks=True)
    r = strict.post("/api/upload", {"file": SimpleUploadedFile("offsite.md", NOTE)})
    assert r.status_code == 403
    strict.get("/")  # the page sets the CSRF cookie
    token = strict.cookies["csrftoken"].value
    r = strict.post("/api/upload", {"file": SimpleUploadedFile("offsite.md", NOTE)}, HTTP_X_CSRFTOKEN=token)
    assert r.status_code == 201


@pytest.mark.parametrize("name, data, status, message", [
    ("evil.exe", b"MZ", 400, "Unsupported file type"),
    ("empty.md", b"", 400, "empty"),
])
def test_bad_uploads_are_rejected_clearly(web, app, name, data, status, message):
    r = upload(web, name, data)
    assert r.status_code == status and message in r.json()["error"]


def test_upload_without_a_file_and_oversized_uploads(web, app, monkeypatch):
    from django.conf import settings
    assert web.post("/api/upload", {}).status_code == 400
    monkeypatch.setattr(settings, "UPLOAD_MAX_BYTES", 50)
    r = upload(web, data=b"x" * 500_000)
    assert r.status_code == 413 and "larger than" in r.json()["error"]
    assert upload(web, data=b"x" * 60).status_code == 413


def test_upload_is_rate_limited(web, app, monkeypatch):
    from django.conf import settings
    monkeypatch.setattr(settings, "UPLOAD_RATE_LIMIT_PER_MIN", 1)
    assert upload(web).status_code == 201
    assert upload(web, "other.md").status_code == 429


def test_removing_an_upload_takes_it_out_of_search(web, app):
    doc_id = upload(web).json()["doc_id"]
    assert web.post(f"/api/uploads/{doc_id}/delete").status_code == 200
    assert doc_id not in services.get_index().docs
    assert web.get(f"/source/{doc_id}/").status_code == 404
    assert web.post(f"/api/uploads/{doc_id}/delete").status_code == 404
    assert web.post("/api/uploads/hr-leave-policy-v2/delete").status_code == 404  # handbook files can't be removed
    assert "hr-leave-policy-v2" in services.get_index().docs


def test_an_upload_from_another_worker_is_picked_up(web, app):
    """Each worker process has its own index; a change to the upload folder rebuilds it on the next request."""
    doc, _ = services.upload_store().add("offsite.md", NOTE, date(2026, 11, 17))
    assert doc.doc_id in services.get_index().docs


def test_page_has_the_upload_control(web, app):
    page = web.get("/").content.decode()
    assert 'id="upload-btn"' in page and 'id="upload-file"' in page and ".docx" in page
    js = open("webapp/handbook/static/handbook/app.js").read()
    assert 'el("a", "cite"' in js and "X-CSRFToken" in js  # [n] markers become links; uploads send the token


def test_rate_limit_uses_the_proxy_address_only_when_trusted(web, app, monkeypatch):
    from django.conf import settings
    monkeypatch.setattr(settings, "ASK_RATE_LIMIT_PER_MIN", 1)
    spoof = {"HTTP_X_FORWARDED_FOR": "1.1.1.1"}
    assert web.post("/api/ask", data=json.dumps({"question": "a"}), content_type="application/json", **spoof).status_code == 200
    # without TRUST_PROXY the header is ignored: a new fake address does not reset the limit
    r = web.post("/api/ask", data=json.dumps({"question": "a"}), content_type="application/json",
                 HTTP_X_FORWARDED_FOR="2.2.2.2")
    assert r.status_code == 429
    monkeypatch.setattr(settings, "TRUST_PROXY", True)
    views._hits.clear()
    for client in ("9.9.9.9, 3.3.3.3", "9.9.9.9, 4.4.4.4"):  # the proxy's own (last) entry decides
        r = web.post("/api/ask", data=json.dumps({"question": "a"}), content_type="application/json",
                     HTTP_X_FORWARDED_FOR=client)
        assert r.status_code == 200


def test_health_reports_the_vector_store(web, app):
    assert web.get("/api/health").json()["vector_store"] == "none (keyword search only)"


def test_answer_says_which_corrected_words_were_also_searched(web, app):
    body = ask(web, "ohw mny anual leves do i get?").json()
    assert body["search_corrections"] == [{"typed": "anual", "searched": "annual"}, {"typed": "leves", "searched": "leave"}]
    assert body["question"] == "ohw mny anual leves do i get?"
    js = open("webapp/handbook/static/handbook/app.js").read()
    assert "also searched for" in js
