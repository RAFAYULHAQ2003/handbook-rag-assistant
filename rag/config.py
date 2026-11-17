import os
from datetime import date


def reference_date(value: str | date | None = None) -> date:
    if isinstance(value, date):
        return value
    raw = value or os.environ.get("REFERENCE_DATE", "").strip()
    if not raw:
        return date.today()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"reference date must be YYYY-MM-DD, got {raw!r}") from exc
