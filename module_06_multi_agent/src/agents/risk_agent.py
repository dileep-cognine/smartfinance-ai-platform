"""Regulatory-risk specialist."""

from __future__ import annotations

from typing import Any

from ..tools import search_regulatory_filings


class RiskAssessmentAgent:
    name = "RiskAssessment"

    def run(self, ticker: str) -> dict[str, Any]:
        return search_regulatory_filings(
            f"What are the material regulatory and financial risks for {ticker}?"
        )
