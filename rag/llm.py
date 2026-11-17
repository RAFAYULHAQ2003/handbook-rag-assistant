import hashlib
import json
import logging
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "llm"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


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


class OpenRouterClient:
    """Chat completions through OpenRouter (OpenAI-compatible), returning parsed JSON.

    - structured output: the reply must match `schema`; only hosts that enforce it are used
    - a reply cut off by the token limit, or that is not valid JSON, is an error, never an answer
    - retries at most `max_retries` times, with a pause, on rate limits, server errors and timeouts
    - identical requests are answered from a disk cache, so re-running an eval costs nothing
    - `max_cost_usd` stops all further calls once that much has been spent by this client
    No temperature is sent: the default model does not accept one on any host.
    max_tokens is generous because the model's hidden reasoning counts toward it (a 90-word
    answer used ~800 tokens); a lower cap cut answers off mid-JSON.
    """

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 60.0, max_retries: int = 2, cache_dir: Path | None = DEFAULT_CACHE_DIR,
                 max_cost_usd: float | None = None):
        missing = [name for name, value in (("OPENROUTER_API_KEY", api_key), ("LLM_MODEL", model),
                                            ("LLM_BASE_URL", base_url)) if not value]
        if missing:
            raise LLMError(f"missing setting(s): {', '.join(missing)} (see .env.example)")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.max_cost_usd = max_cost_usd
        self.spent_usd = 0.0
        self.calls = 0  # paid calls, excluding cache hits

    @classmethod
    def from_env(cls, **overrides) -> "OpenRouterClient":
        """Build from OPENROUTER_API_KEY, LLM_MODEL, LLM_BASE_URL and LLM_MAX_COST_USD.

        There are no built-in fallbacks: a missing setting is an error, so the app can
        never silently call a model nobody chose.
        """
        cap = os.environ.get("LLM_MAX_COST_USD", "").strip()
        try:
            max_cost = float(cap) if cap else None
        except ValueError as exc:
            raise LLMError(f"LLM_MAX_COST_USD must be a number, got {cap!r}") from exc
        settings = {
            "api_key": os.environ.get("OPENROUTER_API_KEY", "").strip(),
            "model": os.environ.get("LLM_MODEL", "").strip(),
            "base_url": os.environ.get("LLM_BASE_URL", "").strip(),
            "max_cost_usd": max_cost,
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
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}},
            "provider": {"require_parameters": True},
            "usage": {"include": True},
        }
        path = self._cache_path(payload)
        if path is not None and path.exists():
            try:
                saved = json.loads(path.read_text())
                return Completion(**{**saved, "cached": True, "latency_ms": 0.0})
            except (json.JSONDecodeError, TypeError):
                log.warning("ignoring unreadable cache entry %s", path.name)

        if self.max_cost_usd is not None and self.spent_usd >= self.max_cost_usd:
            raise BudgetExceeded(f"spending cap of ${self.max_cost_usd:.2f} reached")

        start = time.perf_counter()
        data = self._post(payload)
        latency = (time.perf_counter() - start) * 1000
        self.calls += 1
        usage = data.get("usage") or {}
        cost = float(usage.get("cost") or 0.0)
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
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(min(2 ** attempt, 8))
            request = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in RETRY_STATUS:
                    break  # bad request, bad key, no credit: retrying will not help
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
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
