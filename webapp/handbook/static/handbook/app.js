// Handbook Assistant page. Plain JavaScript, no dependencies.
// Every piece of text from the server is set with textContent, never innerHTML.
(function () {
  "use strict";

  var cfg = window.HANDBOOK;
  var form = document.getElementById("ask-form");
  var input = document.getElementById("question");
  var send = document.getElementById("send");
  var thread = document.getElementById("thread");
  var intro = document.getElementById("intro");
  var asOf = document.getElementById("as-of");
  var statusPill = document.getElementById("status");
  var template = document.getElementById("tpl-turn");
  var newChat = document.getElementById("new-chat");
  var busy = false;

  // ---------- conversation memory (this browser only), so leaving for a source and coming back keeps it ----------
  var STORE = "handbook-conversation-v1";
  var KEEP = 30;
  var MAX_AGE_MS = 7 * 24 * 3600 * 1000;
  function loadTurns() {
    try {
      var turns = JSON.parse(localStorage.getItem(STORE) || "[]");
      var cutoff = Date.now() - MAX_AGE_MS;
      return Array.isArray(turns) ? turns.filter(function (t) { return t && t.at > cutoff && t.answer; }) : [];
    } catch (e) { return []; }
  }
  function saveTurn(question, date, answer) {
    try {
      var turns = loadTurns();
      turns.push({ question: question, date: date, answer: answer, at: Date.now() });
      localStorage.setItem(STORE, JSON.stringify(turns.slice(-KEEP)));
    } catch (e) { /* storage full or disabled: the conversation just won't be kept */ }
  }
  // Earlier turns of this conversation, sent with each question so follow-ups work ("which one lives in Austin?").
  var conversation = [];
  function remember(question, answer) {
    conversation.push({ question: question, answer: answer });
    conversation = conversation.slice(-3);
  }
  newChat.addEventListener("click", function () {
    conversation = [];
    try { localStorage.removeItem(STORE); } catch (e) { /* ignore */ }
    thread.textContent = "";
    intro.classList.remove("compact");
    newChat.hidden = true;
    input.focus();
  });

  // ---------- status + warm-up: loads the index while the user is typing ----------
  function setStatus(kind, text) {
    statusPill.className = "status-pill " + kind;
    statusPill.querySelector(".status-text").textContent = text;
  }
  setStatus("loading", "Loading handbook…");
  fetch(cfg.health, { headers: { Accept: "application/json" } })
    .then(function (r) { return r.json(); })
    .then(function (h) {
      if (h.status === "ok") setStatus("ready", "Ready · " + h.documents + " documents");
      else setStatus("degraded", "Answers unavailable: model not configured");
    })
    .catch(function () { setStatus("degraded", "Offline"); });

  // ---------- composer ----------
  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  document.querySelectorAll("[data-example]").forEach(function (chip) {
    chip.addEventListener("click", function () {
      input.value = chip.getAttribute("data-example");
      autosize();
      form.requestSubmit();
    });
  });
  document.getElementById("as-of-clear").addEventListener("click", function () { asOf.value = ""; });

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var question = input.value.trim();
    if (!question || busy) return;
    if (question.length > cfg.maxChars) question = question.slice(0, cfg.maxChars);
    ask(question, asOf.value || null);
    input.value = "";
    autosize();
  });

  // ---------- one question ----------
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function prettyDate(iso) {
    if (!iso) return "";
    var d = new Date(iso + "T12:00:00");
    return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
  }

  function ask(question, date) {
    busy = true;
    send.disabled = true;
    intro.classList.add("compact");

    var turn = addTurn(question, date);
    turn.scrollIntoView({ behavior: "smooth", block: "start" });

    var steps = {
      search: turn.querySelector('[data-step="search"]'),
      read: turn.querySelector('[data-step="read"]'),
      write: turn.querySelector('[data-step="write"]')
    };
    steps.search.classList.add("active");
    var started = Date.now();
    var timer = null;
    var controller = new AbortController();
    var cancelled = false;
    // the server gives up at about 90 s; the page never waits longer than 2 minutes
    var hardStop = setTimeout(function () { controller.abort(); }, 120000);
    turn.querySelector(".cancel").addEventListener("click", function () {
      cancelled = true;
      controller.abort();
    });

    function done() {
      busy = false;
      send.disabled = false;
      if (timer) clearInterval(timer);
      clearTimeout(hardStop);
      input.focus();
    }

    fetch(cfg.ask, {
      method: "POST",
      signal: controller.signal,
      headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
      body: JSON.stringify({ question: question, as_of: date, history: conversation.slice() })
    })
      .then(function (response) {
        if (!response.ok) {
          return response.json().catch(function () { return {}; }).then(function (body) {
            throw new Error(body.error || "The assistant could not answer (HTTP " + response.status + ").");
          });
        }
        return readLines(response, function (event) {
          if (event.type === "retrieval") {
            steps.search.classList.replace("active", "done");
            steps.read.classList.add("done");
            var n = event.context.length;
            steps.read.querySelector(".read-text").textContent =
              n ? "Reading " + n + " handbook document" + (n === 1 ? "" : "s") : "No matching documents found";
            var list = turn.querySelector(".reading");
            event.context.slice(0, 8).forEach(function (c) { list.appendChild(el("li", null, c.title)); });
            if (n > 8) list.appendChild(el("li", null, "+" + (n - 8) + " more"));
            steps.write.classList.add("active");
            var label = steps.write.querySelector("span:last-child");
            var slowNote = turn.querySelector(".slow-note");
            timer = setInterval(function () {
              var seconds = Math.round((Date.now() - started) / 1000);
              label.textContent = "Writing the answer and checking quotes · " + seconds + "s";
              if (seconds >= 20) slowNote.hidden = false;
            }, 1000);
          } else if (event.type === "answer") {
            showAnswer(turn, event, Date.now() - started);
            if (!(event.error && event.answerable === null) && !event.small_talk) {
              saveTurn(question, date, event);
              remember(question, event.answer);
            }
          } else if (event.type === "error") {
            showError(turn, event.error);
          }
        });
      })
      .catch(function (err) {
        if (err.name === "AbortError") {
          showError(turn, cancelled ? "Cancelled. Ask again whenever you like."
                                    : "This took too long. The model service is slow right now; please try again.");
        } else {
          showError(turn, err.message || "Network error. Please check your connection and try again.");
        }
      })
      .then(done);
  }

  function addTurn(question, date) {
    var turn = template.content.firstElementChild.cloneNode(true);
    turn.querySelector(".question p").textContent = question;
    if (date) turn.querySelector(".as-of-label").textContent = "Answer as of " + prettyDate(date);
    thread.appendChild(turn);
    newChat.hidden = false;
    return turn;
  }

  // Read a newline-delimited JSON stream, calling onEvent for each complete line.
  function readLines(response, onEvent) {
    if (!response.body || !response.body.getReader) {
      return response.text().then(function (text) {
        text.split("\n").filter(Boolean).forEach(function (line) { onEvent(JSON.parse(line)); });
      });
    }
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    function pump() {
      return reader.read().then(function (chunk) {
        if (chunk.done) {
          if (buffer.trim()) onEvent(JSON.parse(buffer));
          return;
        }
        buffer += decoder.decode(chunk.value, { stream: true });
        var lines = buffer.split("\n");
        buffer = lines.pop();
        lines.filter(Boolean).forEach(function (line) { onEvent(JSON.parse(line)); });
        return pump();
      });
    }
    return pump();
  }

  // ---------- rendering an answer ----------
  function badge(cls, text) { return el("span", "badge " + cls, text); }

  // The model sometimes fills the yes/no field for other questions ("What is the password length?" -> "yes").
  // Only show a Yes/No badge when some sentence of the question is a yes/no question.
  var YES_NO = /(^|[.?!]\s+)(can|could|do|does|did|is|are|was|were|will|would|should|shall|may|might|must|has|have|had|am)\b/i;
  function isYesNoQuestion(question) { return YES_NO.test((question || "").trim()); }

  // The model sometimes repeats its sources inside the answer ("Citations: hr-... 'quote'"). They are
  // already shown as source cards, so cut that tail off when the answer has cards.
  var CITATION_TAIL = /(^|\n|\s)(citations?|sources?)\s*:\s/i;
  function cleanAnswer(text, hasCitations) {
    text = text || "";
    if (hasCitations) {
      var m = CITATION_TAIL.exec(text);
      if (m && m.index > text.length * 0.3) text = text.slice(0, m.index);
    }
    return text.trim();
  }

  // Show **bold** as bold and [n] as a link to source n, without ever parsing HTML:
  // text nodes, <strong> and <a> elements only. The server has already renumbered the markers
  // so each one points at a verified quote; one without a matching source stays plain text.
  function appendWithMarkers(parent, text, citations, turn) {
    text.split(/(\[\d{1,2}\])/g).forEach(function (part) {
      if (!part) return;
      var m = /^\[(\d{1,2})\]$/.exec(part);
      var c = m && citations[Number(m[1]) - 1];
      if (!c) { parent.appendChild(document.createTextNode(part)); return; }
      var link = el("a", "cite", m[1]);
      link.href = c.url;
      link.target = "_blank";
      link.rel = "noopener";
      link.title = c.title + ": \u201c" + c.quote + "\u201d";
      link.setAttribute("aria-label", "Source " + m[1] + ": " + c.title);
      link.addEventListener("mouseenter", function () { flashSource(turn, Number(m[1])); });
      link.addEventListener("focus", function () { flashSource(turn, Number(m[1])); });
      parent.appendChild(link);
    });
  }
  function flashSource(turn, n) {
    var item = turn && turn.querySelectorAll(".sources .source")[n - 1];
    if (!item) return;
    item.classList.add("flash");
    setTimeout(function () { item.classList.remove("flash"); }, 1200);
  }
  // **bold** and `code`; everything else is plain text (line breaks and "- " lists survive via pre-wrap)
  function renderInline(node, text, citations, turn) {
    node.textContent = "";
    var parts = text.split(/(\*\*.+?\*\*|`[^`\n]+`)/g);
    parts.forEach(function (part) {
      if (!part) return;
      if (/^\*\*.+\*\*$/.test(part)) appendWithMarkers(node.appendChild(el("strong")), part.slice(2, -2), citations || [], turn);
      else if (/^`[^`]+`$/.test(part)) node.appendChild(el("code", null, part.slice(1, -1)));
      else appendWithMarkers(node, part, citations || [], turn);
    });
  }
  function plainText(text) { return text.replace(/\*\*(.+?)\*\*/g, "$1").replace(/`([^`\n]+)`/g, "$1"); }

  function showAnswer(turn, a, elapsedMs) {
    var progress = turn.querySelector(".progress");
    var result = turn.querySelector(".result");
    progress.hidden = true;
    result.hidden = false;

    if (a.small_talk) {
      turn.querySelector(".answer-text").textContent = a.answer;
      turn.querySelector(".meta").textContent = "";
      turn.querySelector(".copy").hidden = true;
      return;
    }

    if (a.error && a.answerable === null) {
      result.classList.add("error");
      turn.querySelector(".answer-text").textContent = a.answer;
      turn.querySelector(".meta").textContent = "The model service did not respond. Please try again in a moment.";
      return;
    }

    var badges = turn.querySelector(".badges");
    if (isYesNoQuestion(a.question)) {
      if (a.verdict === "yes") badges.appendChild(badge("badge-yes", "Yes"));
      if (a.verdict === "no") badges.appendChild(badge("badge-no", "No"));
    }
    if (a.needs_clarification) badges.appendChild(badge("badge-info", "Which one do you mean?"));
    if (a.answerable === false) badges.appendChild(badge("badge-warn", "Not fully covered by the handbook"));
    if (a.unsupported) badges.appendChild(badge("badge-warn", "No verified source: check with the policy owner"));

    var shown = cleanAnswer(a.answer, a.citations.length > 0);
    renderInline(turn.querySelector(".answer-text"), shown, a.citations, turn);

    if (a.citations.length) {
      turn.querySelector(".sources-block").hidden = false;
      var list = turn.querySelector(".sources");
      a.citations.forEach(function (c) {
        var item = el("li");
        var link = el("a", "source");
        link.href = c.url;
        link.target = "_blank";
        link.rel = "noopener";
        link.title = "Open " + c.title + " with this passage highlighted";
        var head = el("span", "source-head");
        head.appendChild(el("span", null, c.title));
        if (c.state !== "in_force") head.appendChild(el("span", "state state-" + c.state, c.state_label));
        if (c.uploaded) head.appendChild(el("span", "state state-uploaded", "uploaded, not official"));
        head.appendChild(el("span", "source-open", "Open source ↗"));
        link.appendChild(head);
        link.appendChild(el("p", "source-quote", c.quote));
        item.appendChild(link);
        list.appendChild(item);
      });
    }

    var docs = a.context.length;
    var parts = ["Searched " + docs + " document" + (docs === 1 ? "" : "s")];
    if (a.reference_date) parts.push("as of " + prettyDate(a.reference_date));
    if (elapsedMs === null) parts.push("earlier");
    else parts.push(a.cached ? "instant (cached)" : (elapsedMs / 1000).toFixed(1) + "s");
    if (a.used_fallback) parts.push("answered by the backup model (main model busy)");
    turn.querySelector(".meta").textContent = parts.join(" · ");

    var copy = turn.querySelector(".copy");
    copy.addEventListener("click", function () {
      var text = plainText(shown) + (a.citations.length ? "\n\nSources:\n" + a.citations.map(function (c) {
        return "[" + c.n + "] " + c.title + ": \"" + c.quote + "\"";
      }).join("\n") : "");
      navigator.clipboard.writeText(text).then(function () {
        copy.textContent = "Copied";
        setTimeout(function () { copy.textContent = "Copy answer"; }, 1500);
      });
    });
  }

  function showError(turn, message) {
    turn.querySelector(".progress").hidden = true;
    var result = turn.querySelector(".result");
    result.hidden = false;
    result.classList.add("error");
    turn.querySelector(".answer-text").textContent = message;
    turn.querySelector(".copy").hidden = true;
  }

  // ---------- uploads: add a document and ask about it straight away ----------
  var uploadBtn = document.getElementById("upload-btn");
  var uploadFile = document.getElementById("upload-file");
  var uploadsBox = document.getElementById("uploads");
  var uploadStatus = document.getElementById("upload-status");

  function csrfToken() {
    var m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
    return m ? decodeURIComponent(m[1]) : "";
  }
  function showUploadStatus(text, isError) {
    uploadStatus.textContent = text;
    uploadStatus.className = "upload-status" + (isError ? " error" : "");
    uploadStatus.hidden = !text;
  }
  function renderUploads(docs) {
    uploadsBox.textContent = "";
    docs.forEach(function (d) {
      var chip = el("span", "upload-chip");
      var link = el("a", null, d.title);
      link.href = d.url;
      link.title = "Open " + d.title;
      var remove = el("button", null, "\u00d7");
      remove.type = "button";
      remove.title = "Remove " + d.title;
      remove.setAttribute("aria-label", "Remove " + d.title);
      remove.addEventListener("click", function () { removeUpload(d); });
      chip.appendChild(el("span", null, "\ud83d\udcc4"));
      chip.appendChild(link);
      chip.appendChild(remove);
      uploadsBox.appendChild(chip);
    });
    uploadsBox.hidden = docs.length === 0;
  }
  function refreshUploads() {
    return fetch(cfg.uploads, { headers: { Accept: "application/json" } })
      .then(function (r) { return r.ok ? r.json() : { documents: [] }; })
      .then(function (body) { renderUploads(body.documents || []); })
      .catch(function () { /* the list is a convenience; asking still works */ });
  }
  function postForm(url, body) {
    return fetch(url, { method: "POST", headers: { "X-CSRFToken": csrfToken(), Accept: "application/json" },
                        body: body, credentials: "same-origin" })
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (data) {
          if (!r.ok) throw new Error(data.error || "The request failed (HTTP " + r.status + ").");
          return data;
        });
      });
  }
  function removeUpload(d) {
    postForm(cfg.uploadDelete.replace("__id__", encodeURIComponent(d.doc_id)), new FormData())
      .then(function () { showUploadStatus("Removed \u201c" + d.title + "\u201d.", false); return refreshUploads(); })
      .catch(function (err) { showUploadStatus(err.message, true); });
  }
  uploadBtn.addEventListener("click", function () { uploadFile.click(); });
  uploadFile.addEventListener("change", function () {
    var file = uploadFile.files && uploadFile.files[0];
    uploadFile.value = "";
    if (!file) return;
    if (file.size > cfg.uploadMaxMb * 1000000) {
      showUploadStatus("\u201c" + file.name + "\u201d is larger than " + cfg.uploadMaxMb + " MB.", true);
      return;
    }
    var body = new FormData();
    body.append("file", file);
    uploadBtn.disabled = true;
    showUploadStatus("Reading \u201c" + file.name + "\u201d\u2026", false);
    postForm(cfg.upload, body)
      .then(function (d) {
        showUploadStatus((d.created ? "Added \u201c" : "Already added: \u201c") + d.title +
          "\u201d. Ask about it now; answers will cite it and mark it as not official.", false);
        input.placeholder = "Ask about \u201c" + d.title + "\u201d\u2026";
        input.focus();
        return refreshUploads();
      })
      .catch(function (err) { showUploadStatus(err.message, true); })
      .then(function () { uploadBtn.disabled = false; });
  });
  refreshUploads();

  // ---------- start-up: runs last, so every helper above is defined ----------
  // Restore earlier answers from this browser.
  var earlier = loadTurns();
  if (earlier.length) {
    intro.classList.add("compact");
    earlier.forEach(function (t) {
      showAnswer(addTurn(t.question, t.date), t.answer, null);
      remember(t.question, t.answer.answer || "");
    });
    thread.lastElementChild.scrollIntoView({ block: "end" });
  }

  // Shareable links: /?q=question[&as_of=2027-01-15] asks straight away.
  var params = new URLSearchParams(window.location.search);
  if (params.get("q")) {
    input.value = params.get("q").slice(0, cfg.maxChars);
    if (/^\d{4}-\d{2}-\d{2}$/.test(params.get("as_of") || "")) asOf.value = params.get("as_of");
    autosize();
    setTimeout(function () { form.requestSubmit(); }, 0);
  }
})();
