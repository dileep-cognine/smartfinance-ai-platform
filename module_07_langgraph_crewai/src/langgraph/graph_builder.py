"""Stateful LangGraph research workflow with revision and escalation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TypedDict

import httpx
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph


class GraphState(TypedDict, total=False):
    ticker: str
    stock_data: dict
    news_sentiment: dict
    risk_findings: dict
    draft_report: str
    review_comments: str
    final_report: str
    status: str
    revision_count: int


def fetch_data(state: GraphState) -> GraphState:
    ticker = state["ticker"]
    return {
        "stock_data": {"ticker": ticker, "source": "Module 6 DataFetcher"},
        "status": "data_fetched",
    }


def analyze_news(state: GraphState) -> GraphState:
    return {
        "news_sentiment": {
            "ticker": state["ticker"],
            "source": "Module 3 FinBERT",
            "sentiment": "pending_model_inference",
        },
        "status": "news_analyzed",
    }


def assess_risk(state: GraphState) -> GraphState:
    try:
        response = httpx.post(
            "http://rag_api:8000/query",
            json={
                "question": f"Material risks for {state['ticker']}",
                "session_id": f"graph-{state['ticker']}",
            },
            timeout=30,
        )
        response.raise_for_status()
        finding = response.json()
    except httpx.HTTPError as exc:
        finding = {"error": str(exc)}
    return {"risk_findings": finding, "status": "risk_assessed"}


def draft_report(state: GraphState) -> GraphState:
    revision = state.get("revision_count", 0)
    comments = state.get("review_comments", "")
    draft = (
        f"# Research Report: {state['ticker']}\n\n"
        f"## Market Data\n{json.dumps(state.get('stock_data', {}), indent=2)}\n\n"
        f"## News\n{json.dumps(state.get('news_sentiment', {}), indent=2)}\n\n"
        f"## Risks\n{json.dumps(state.get('risk_findings', {}), indent=2)}\n\n"
    )
    if comments and "APPROVED" not in comments.upper():
        draft += f"## Revision Response\nAddressed reviewer comments: {comments}\n"
    return {"draft_report": draft, "revision_count": revision + 1, "status": "drafted"}


def review_report(state: GraphState) -> GraphState:
    comments = state.get("review_comments", "")
    if not comments:
        comments = "APPROVED" if len(state.get("draft_report", "")) > 200 else "Add evidence."
    return {"review_comments": comments, "status": "reviewed"}


def review_route(state: GraphState) -> str:
    if "APPROVED" in state.get("review_comments", "").upper():
        return "finalize"
    if state.get("revision_count", 0) >= 3:
        return "human_review"
    return "revise"


def human_review(state: GraphState) -> GraphState:
    return {"status": "human_review_required"}


def finalize_report(state: GraphState) -> GraphState:
    return {"final_report": state["draft_report"], "status": "completed"}


def build_graph(database_path: Path):
    workflow = StateGraph(GraphState)
    workflow.add_node("fetch_data", fetch_data)
    workflow.add_node("analyze_news", analyze_news)
    workflow.add_node("assess_risk", assess_risk)
    workflow.add_node("draft_report", draft_report)
    workflow.add_node("review_report", review_report)
    workflow.add_node("human_review", human_review)
    workflow.add_node("finalize_report", finalize_report)
    workflow.set_entry_point("fetch_data")
    workflow.add_edge("fetch_data", "analyze_news")
    workflow.add_edge("analyze_news", "assess_risk")
    workflow.add_edge("assess_risk", "draft_report")
    workflow.add_edge("draft_report", "review_report")
    workflow.add_conditional_edges(
        "review_report",
        review_route,
        {
            "finalize": "finalize_report",
            "revise": "draft_report",
            "human_review": "human_review",
        },
    )
    workflow.add_edge("finalize_report", END)
    workflow.add_edge("human_review", END)
    connection = sqlite3.connect(database_path, check_same_thread=False)
    return workflow.compile(checkpointer=SqliteSaver(connection))


def run_or_resume(graph, ticker: str, thread_id: str) -> GraphState:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if snapshot.values:
        return graph.invoke(None, config)
    return graph.invoke(
        {"ticker": ticker.upper(), "revision_count": 0, "status": "started"},
        config,
    )


def save_mermaid(graph, output_path: Path) -> None:
    output_path.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
