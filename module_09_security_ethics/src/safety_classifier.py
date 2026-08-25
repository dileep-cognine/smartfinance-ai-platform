"""Secondary output safety classifier using Groq free-tier LLM with deterministic offline fallback."""

from __future__ import annotations

import json
import os

from .guardrails import inspect_output


def classify_output(text: str) -> dict:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        result = inspect_output(text)
        return {
            "safe": result.allowed,
            "categories": result.reasons,
            "classifier": "deterministic_fallback",
        }

    try:
        from groq import Groq

        model = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify financial assistant output. Return JSON with 'safe' (boolean) "
                        "and 'categories' (array of strings). Unsafe categories include PII, personalized "
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
            "classifier": "groq_llm",
        }
    except Exception:
        result = inspect_output(text)
        return {
            "safe": result.allowed,
            "categories": result.reasons,
            "classifier": "deterministic_fallback",
        }
