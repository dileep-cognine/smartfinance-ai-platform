"""Security, privacy, and lightweight explanation API."""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .guardrails import anonymize_pii, inspect_input, inspect_output
from .safety_classifier import classify_output

app = FastAPI(title="SmartFinance Security and XAI API", version="0.1.0")


class GuardRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    direction: str = Field(default="input", pattern=r"^(input|output)$")


class GuardResponse(BaseModel):
    allowed: bool
    text: str
    reasons: list[str]


class ExplanationRequest(BaseModel):
    contributions: dict[str, float]
    decision: str = Field(pattern=r"^(approved|denied)$")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/guard", response_model=GuardResponse)
def guard(request: GuardRequest) -> GuardResponse:
    result = (
        inspect_input(request.text, int(os.getenv("MAX_PROMPT_LENGTH", "4000")))
        if request.direction == "input"
        else inspect_output(request.text)
    )
    if request.direction == "output" and result.allowed:
        secondary = classify_output(request.text)
        if not secondary["safe"]:
            result = type(result)(
                allowed=False,
                sanitized_text=result.sanitized_text,
                reasons=list(secondary["categories"]),
            )
    return GuardResponse(
        allowed=result.allowed,
        text=result.sanitized_text,
        reasons=result.reasons,
    )


@app.post("/anonymize")
def anonymize(request: GuardRequest) -> dict[str, str]:
    return {"text": anonymize_pii(request.text)}


@app.post("/explain")
def explain(request: ExplanationRequest) -> dict[str, str]:
    ordered = sorted(request.contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    primary = ordered[:3]
    direction = "increased" if request.decision == "denied" else "reduced"
    features = ", ".join(name.replace("_", " ") for name, _ in primary)
    return {
        "explanation": (
            f"Your application was {request.decision}. The strongest factors that "
            f"{direction} estimated credit risk were: {features}."
        )
    }
