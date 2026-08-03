"""News sentiment specialist."""

from __future__ import annotations

from typing import Any

from ..tools import get_news_sentiment


class NewsAnalysisAgent:
    name = "NewsAnalysis"

    def run(self, ticker: str) -> dict[str, Any]:
        return get_news_sentiment(ticker, 7)
