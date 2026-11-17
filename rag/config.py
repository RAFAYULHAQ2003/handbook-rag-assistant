import os
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TIMEZONE = "America/Chicago"  # headquarters in Austin; core hours are in Central time


def company_timezone() -> ZoneInfo:
    name = os.environ.get("COMPANY_TIMEZONE", "").strip() or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown COMPANY_TIMEZONE {name!r}") from exc


def reference_date(value: str | date | None = None) -> date:
    """Explicit value, else REFERENCE_DATE, else today in the company's time zone (not the server's UTC)."""
    if isinstance(value, date):
        return value
    raw = value or os.environ.get("REFERENCE_DATE", "").strip()
    if not raw:
        return datetime.now(company_timezone()).date()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"reference date must be YYYY-MM-DD, got {raw!r}") from exc
