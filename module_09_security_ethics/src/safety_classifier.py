"""Secondary output classifier with an explicit offline fallback."""

from __future__ import annotations

import json
import os

from openai import OpenAI

from .guardrails import inspect_output


def classify_output(text: str) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        result = inspect_output(text)
        return {
            "safe": result.allowed,
            "categories": result.reasons,
            "classifier": "deterministic_fallback",
        }
    response = OpenAI().chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify financial assistant output. Return JSON with safe boolean "
                    "and categories array. Unsafe categories include PII, personalized "
                    "investment advice, guaranteed returns, prompt leakage, and evasion."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    result = json.loads(response.choices[0].message.content or "{}")
    return {
        "safe": bool(result.get("safe", False)),
        "categories": result.get("categories", ["classifier_parse_failure"]),
        "classifier": "llm",
    }
