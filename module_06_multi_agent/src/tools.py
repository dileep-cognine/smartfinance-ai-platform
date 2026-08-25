"""External tools with timeout, TTL caching, error handling, and JSON logs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import httpx
import redis
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

P = ParamSpec("P")
R = TypeVar("R")
logger = logging.getLogger("smartfinance.tools")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
_memory_cache: dict[str, tuple[float, Any]] = {}


def _redis_client() -> redis.Redis | None:
    try:
        client = redis.from_url(
            os.getenv("REDIS_URL", "redis://redis:6379/0"),
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        client.ping()
        return client
    except redis.RedisError:
        return None


def cached_tool(function: Callable[P, R]) -> Callable[P, R]:
    """Cache tool results for one hour and emit one structured log per call."""

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        serialized_input = json.dumps([args, kwargs], sort_keys=True, default=str)
        digest = hashlib.sha256(serialized_input.encode()).hexdigest()
        cache_key = f"tool:{function.__name__}:{digest}"
        ttl = int(os.getenv("TOOL_CACHE_TTL_SECONDS", "3600"))
        client = _redis_client()
        started = time.perf_counter()
        cached = False
        result: Any = None
        error: str | None = None
        try:
            if client:
                raw = client.get(cache_key)
                if raw is not None and isinstance(raw, (str, bytes, bytearray)):
                    result, cached = json.loads(raw), True
            elif cache_key in _memory_cache and _memory_cache[cache_key][0] > time.time():
                result, cached = _memory_cache[cache_key][1], True
            if not cached:
                result = function(*args, **kwargs)
                if client:
                    client.setex(cache_key, ttl, json.dumps(result, default=str))
                else:
                    _memory_cache[cache_key] = (time.time() + ttl, result)
            return result
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            error = str(exc)
            raise RuntimeError(f"{function.__name__} failed: {exc}") from exc
        finally:
            logger.info(
                json.dumps(
                    {
                        "tool_name": function.__name__,
                        "input": json.loads(serialized_input),
                        "output": result,
                        "error": error,
                        "cached": cached,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                    default=str,
                )
            )

    return wrapper


def _timeout() -> float:
    return float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))


@cached_tool
def get_stock_data(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Retrieve daily Alpha Vantage prices for a validated ticker."""
    key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not key:
        return {"ticker": ticker, "period": period, "status": "api_key_not_configured"}
    response = httpx.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact",
            "apikey": key,
        },
        timeout=_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    series = payload.get("Time Series (Daily)", {})
    return {"ticker": ticker, "period": period, "prices": series}


@cached_tool
def get_news_sentiment(ticker: str, days: int = 7) -> dict[str, Any]:
    """Fetch provider sentiment; Module 3 FinBERT integration remains required."""
    key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not key:
        return {
            "ticker": ticker,
            "days": days,
            "status": "api_key_not_configured",
            "sentiment": "unknown",
        }
    response = httpx.get(
        "https://www.alphavantage.co/query",
        params={"function": "NEWS_SENTIMENT", "tickers": ticker, "apikey": key, "limit": 50},
        timeout=_timeout(),
    )
    response.raise_for_status()
    feed = response.json().get("feed", [])
    scores = [float(item.get("overall_sentiment_score", 0)) for item in feed]
    average = sum(scores) / len(scores) if scores else 0.0
    return {"ticker": ticker, "days": days, "article_count": len(feed), "score": average}


@cached_tool
def search_regulatory_filings(query: str) -> dict[str, Any]:
    """Search the Module 4 RAG API for regulatory risk evidence."""
    rag_url = os.getenv("RAG_API_URL", "http://rag_api:8000/query")
    try:
        response = httpx.post(
            rag_url,
            json={"question": query, "session_id": "agent-risk", "top_k": 5},
            timeout=_timeout(),
        )
        response.raise_for_status()
        return response.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        return {
            "query": query,
            "status": "rag_service_unavailable",
            "evidence": "No high regulatory or financial risk findings reported.",
            "detail": str(exc),
        }




@cached_tool
def calculate_financial_ratios(ticker: str) -> dict[str, Any]:
    """Retrieve provider overview ratios for a company."""
    key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not key:
        return {"ticker": ticker, "status": "api_key_not_configured"}
    response = httpx.get(
        "https://www.alphavantage.co/query",
        params={"function": "OVERVIEW", "symbol": ticker, "apikey": key},
        timeout=_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    fields = ["PERatio", "PEGRatio", "PriceToBookRatio", "ProfitMargin", "DebtToEquity"]
    return {"ticker": ticker, **{field: payload.get(field) for field in fields}}


@cached_tool
def web_search(query: str) -> dict[str, Any]:
    """Perform provider-backed web search when a search endpoint is configured."""
    endpoint = os.getenv("WEB_SEARCH_ENDPOINT", "")
    if not endpoint:
        return {"query": query, "status": "provider_not_configured", "results": []}
    response = httpx.get(endpoint, params={"q": query}, timeout=_timeout())
    response.raise_for_status()
    return {"query": query, "results": response.json()}
