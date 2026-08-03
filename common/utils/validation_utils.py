from __future__ import annotations

import re

TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def validate_ticker(value: str) -> str:
    normalized = value.strip().upper()
    if not TICKER_PATTERN.fullmatch(normalized):
        raise ValueError("Invalid ticker")
    return normalized


def require_keys(value: dict, keys: set[str]) -> None:
    missing = keys - set(value)
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(sorted(missing))}")
