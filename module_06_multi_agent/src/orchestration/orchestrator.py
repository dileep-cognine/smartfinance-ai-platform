"""Ordered research workflow with retries and a human checkpoint."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..agents.analyst_agent import NewsAnalysisAgent
from ..agents.data_fetcher_agent import DataFetcherAgent
from ..agents.report_agent import ReportWriterAgent
from ..agents.risk_agent import RiskAssessmentAgent
from ..communication.messaging import ResearchState


def _retry(operation: Callable[[], dict[str, Any]], attempts: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except RuntimeError as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"Operation failed after {attempts} attempts") from last_error


class Orchestrator:
    def __init__(self) -> None:
        self.data_fetcher = DataFetcherAgent()
        self.news_analyst = NewsAnalysisAgent()
        self.risk_assessor = RiskAssessmentAgent()
        self.report_writer = ReportWriterAgent()

    @staticmethod
    def _timed(operation: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        value = _retry(operation)
        return value, (time.perf_counter() - started) * 1000

    def start(self, ticker: str) -> ResearchState:
        normalized = ticker.strip().upper()
        state = ResearchState(run_id=str(uuid.uuid4()), ticker=normalized)
        with ThreadPoolExecutor(max_workers=2) as executor:
            data_future = executor.submit(
                self._timed, lambda: self.data_fetcher.run(normalized)
            )
            news_future = executor.submit(
                self._timed, lambda: self.news_analyst.run(normalized)
            )
            state.stock_data, state.timings_ms["DataFetcher"] = data_future.result()
            state.news_sentiment, state.timings_ms["NewsAnalysis"] = news_future.result()
        state.risk_findings, state.timings_ms["RiskAssessment"] = self._timed(
            lambda: self.risk_assessor.run(normalized)
        )
        state.status = "awaiting_confirmation"
        return state

    def finalize(self, state: ResearchState) -> ResearchState:
        started = time.perf_counter()
        state.final_report = self.report_writer.run(state)
        state.timings_ms["ReportWriter"] = (time.perf_counter() - started) * 1000
        state.status = "completed"
        return state
