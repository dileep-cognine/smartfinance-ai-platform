"""Market-data specialist."""

from __future__ import annotations

from typing import Any

from ..tools import calculate_financial_ratios, get_stock_data


class DataFetcherAgent:
    name = "DataFetcher"

    def run(self, ticker: str) -> dict[str, Any]:
        return {
            "prices": get_stock_data(ticker, "1y"),
            "ratios": calculate_financial_ratios(ticker),
        }
