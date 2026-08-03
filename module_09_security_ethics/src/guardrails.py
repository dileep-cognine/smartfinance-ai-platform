"""Input/output guardrails for the financial RAG service."""

from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"reveal\s+(the\s+)?system\s+prompt",
        r"\byou\s+are\s+now\b",
        r"\bdeveloper\s+message\b",
        r"\bjailbreak\b",
        r"\bdo\s+anything\s+now\b",
    ]
]
ADVICE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bguaranteed\s+(return|profit)\b",
        r"\b(buy|sell|short)\s+[A-Z]{1,5}\b",
        r"\binvest\s+all\b",
    ]
]
PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)"),
    "ACCOUNT": re.compile(r"\b(?:account\s*(?:number|no\.?)?\s*[:#-]?\s*)\d{8,17}\b", re.I),
}


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    sanitized_text: str
    reasons: list[str]


def anonymize_pii(text: str) -> str:
    """Replace common structured PII with typed placeholders."""
    result = text
    for label, pattern in PII_PATTERNS.items():
        result = pattern.sub(f"[{label}_REDACTED]", result)
    return result


def inspect_input(text: str, max_length: int = 4000) -> GuardResult:
    """Apply length, injection, and PII controls before retrieval."""
    reasons: list[str] = []
    if len(text) > max_length:
        reasons.append("prompt_too_long")
    if any(pattern.search(text) for pattern in INJECTION_PATTERNS):
        reasons.append("prompt_injection")
    sanitized = anonymize_pii(text[:max_length])
    return GuardResult(not reasons, sanitized, reasons)


def inspect_output(text: str) -> GuardResult:
    """Block PII and directive regulated investment advice."""
    reasons: list[str] = []
    if anonymize_pii(text) != text:
        reasons.append("pii")
    if any(pattern.search(text) for pattern in ADVICE_PATTERNS):
        reasons.append("regulated_financial_advice")
    return GuardResult(not reasons, anonymize_pii(text), reasons)
