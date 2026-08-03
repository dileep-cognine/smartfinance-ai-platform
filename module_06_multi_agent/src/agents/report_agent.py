"""Research brief writer."""

from __future__ import annotations

import json

from ..communication.messaging import ResearchState


class ReportWriterAgent:
    name = "ReportWriter"

    def run(self, state: ResearchState) -> str:
        return (
            f"# Investment Research Brief: {state.ticker}\n\n"
            "## Financial Data\n\n"
            f"```json\n{json.dumps(state.stock_data, indent=2)}\n```\n\n"
            "## News Sentiment\n\n"
            f"```json\n{json.dumps(state.news_sentiment, indent=2)}\n```\n\n"
            "## Regulatory Risk\n\n"
            f"```json\n{json.dumps(state.risk_findings, indent=2)}\n```\n\n"
            "## Limitations\n\n"
            "This report is research support, not personalized investment advice.\n"
        )
