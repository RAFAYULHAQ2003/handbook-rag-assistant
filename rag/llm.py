import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "llm"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
RATE_LIMIT_BACKOFF = (10.0, 20.0, 40.0, 60.0)  # seconds, for 429s that carry no Retry-After


class LLMError(Exception):
    """The model call failed or returned something unusable."""


class BudgetExceeded(LLMError):
    """The spending cap for this client was reached; no further calls are made."""


@dataclass(frozen=True)
class Completion:
    content: dict
    model: str
    provider: str | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    cached: bool


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[dict], schema: dict, max_tokens: int = 4000) -> Completion: ...


class OpenAICompatibleClient:
    """Chat completions from any OpenAI-compatible API (NVIDIA, OpenRouter, ...), returning parsed JSON.

    - structured output: the reply must match `schema` (`response_format: json_schema`)
    - a reply cut off by the token limit, or that is not valid JSON, is an error, never an answer
    - retries rate limits, server errors and timeouts at most `max_retries` times, waiting as long
      as the server asks (Retry-After) or backing off; never retries a bad key or a bad request
    - identical requests are answered from a disk cache, so re-running an eval costs nothing
    - `max_cost_usd` stops all further calls once that much has been spent by this client, when the
      provider reports cost (OpenRouter does; NVIDIA's free tier does not, so cost reads as $0)
      (with parallel workers, calls already in flight may finish, so the cap can be passed slightly)
    - `extra_body` adds provider-specific fields from LLM_EXTRA_BODY, such as NVIDIA's
      {"chat_template_kwargs": {"enable_thinking": false}}
    - OpenRouter-only fields (host routing, cost reporting) are sent only to OpenRouter; other
      providers reject them
    No temperature is sent, so each provider's default is used (Claude Haiku on OpenRouter rejects it).
    max_tokens is generous because hidden reasoning can count toward it; a 1,200 cap cut answers off.
    """

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 120.0, max_retries: int = 4,
                 cache_dir: Path | None = DEFAULT_CACHE_DIR, max_cost_usd: float | None = None,
                 extra_body: dict | None = None):
        missing = [name for name, value in (("LLM_API_KEY", api_key), ("LLM_MODEL", model),
                                            ("LLM_BASE_URL", base_url)) if not value]
        if missing:
            raise LLMError(f"missing setting(s): {', '.join(missing)} (see .env.example)")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.extra_body = dict(extra_body or {})
        self.is_openrouter = "openrouter.ai" in self.base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.max_cost_usd = max_cost_usd
        self.spent_usd = 0.0
        self.calls = 0  # paid calls, excluding cache hits
        self._lock = threading.Lock()  # counters are shared by parallel eval workers and web requests

    @classmethod
    def from_env(cls, **overrides) -> "OpenAICompatibleClient":
        """Build from LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, LLM_EXTRA_BODY and LLM_MAX_COST_USD.

        There are no built-in fallbacks: a missing setting is an error, so the app can
        never silently call a model nobody chose.
        """
        cap = os.environ.get("LLM_MAX_COST_USD", "").strip()
        try:
            max_cost = float(cap) if cap else None
        except ValueError as exc:
            raise LLMError(f"LLM_MAX_COST_USD must be a number, got {cap!r}") from exc
        extra = os.environ.get("LLM_EXTRA_BODY", "").strip()
        try:
            extra_body = json.loads(extra) if extra else None
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM_EXTRA_BODY must be a JSON object, got {extra!r}") from exc
        if extra_body is not None and not isinstance(extra_body, dict):
            raise LLMError("LLM_EXTRA_BODY must be a JSON object, e.g. {\"chat_template_kwargs\": {...}}")
        settings = {
            "api_key": os.environ.get("LLM_API_KEY", "").strip(),
            "model": os.environ.get("LLM_MODEL", "").strip(),
            "base_url": os.environ.get("LLM_BASE_URL", "").strip(),
            "max_cost_usd": max_cost,
            "extra_body": extra_body,
        }
        settings.update(overrides)
        return cls(**settings)

    def _cache_path(self, payload: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def complete(self, messages: list[dict], schema: dict, max_tokens: int = 4000) -> Completion:
        payload = {
            **self.extra_body,  # provider extras first, so they can never replace the fields below
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}},
        }
        if self.is_openrouter:
            payload["provider"] = {"require_parameters": True}  # only hosts that enforce the format
            payload["usage"] = {"include": True}  # report the cost of each call
        path = self._cache_path(payload)
        if path is not None and path.exists():
            try:
                saved = json.loads(path.read_text())
                return Completion(**{**saved, "cached": True, "latency_ms": 0.0})
            except (json.JSONDecodeError, TypeError):
                log.warning("ignoring unreadable cache entry %s", path.name)

        with self._lock:
            if self.max_cost_usd is not None and self.spent_usd >= self.max_cost_usd:
                raise BudgetExceeded(f"spending cap of ${self.max_cost_usd:.2f} reached")

        start = time.perf_counter()
        data = self._post(payload)
        latency = (time.perf_counter() - start) * 1000
        usage = data.get("usage") or {}
        cost = float(usage.get("cost") or 0.0)
        with self._lock:
            self.calls += 1
            self.spent_usd += cost

        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {str(data)[:200]}") from exc
        if choice.get("finish_reason") == "length":
            raise LLMError("the reply was cut off by the token limit")
        try:
            content = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMError(f"the reply was not valid JSON: {str(text)[:200]}") from exc
        if not isinstance(content, dict):
            raise LLMError("the reply was not a JSON object")

        completion = Completion(content, data.get("model") or self.model, data.get("provider"),
                                int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
                                cost, latency, False)
        if path is not None:
            self._save(path, completion)
        return completion

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        last_error = "no attempt made"
        wait = 0.0
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(wait or min(2 ** attempt, 16))
            request = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in RETRY_STATUS:
                    break  # bad request, bad key, no credit: retrying will not help
                wait = _retry_after(exc.headers)
                if not wait and exc.code == 429:
                    # rate limited without a hint: free tiers reset per minute, so short waits just fail again
                    wait = RATE_LIMIT_BACKOFF[min(attempt, len(RATE_LIMIT_BACKOFF) - 1)]
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                wait = 0.0
            except json.JSONDecodeError as exc:
                last_error = f"response was not JSON: {exc}"
        raise LLMError(f"model call failed: {last_error}")

    def _save(self, path: Path, completion: Completion) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {k: v for k, v in completion.__dict__.items() if k not in ("cached", "latency_ms")}
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(record, f)
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def _retry_after(headers) -> float:
    """Seconds the server asked us to wait (Retry-After), capped at 60; 0 if it did not say."""
    try:
        return min(max(float((headers or {}).get("Retry-After", 0)), 0.0), 60.0)
    except (TypeError, ValueError):
        return 0.0
