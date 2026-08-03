"""Typed messages exchanged by research agents."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    run_id: str
    sender: str
    recipient: str
    status: Literal["success", "failed", "awaiting_confirmation"]
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResearchState(BaseModel):
    run_id: str
    ticker: str
    stock_data: dict[str, Any] = Field(default_factory=dict)
    news_sentiment: dict[str, Any] = Field(default_factory=dict)
    risk_findings: dict[str, Any] = Field(default_factory=dict)
    final_report: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    status: Literal["running", "awaiting_confirmation", "completed", "failed"] = "running"
