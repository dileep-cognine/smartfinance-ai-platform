"""CrewAI version of the SmartFinance research team."""

from __future__ import annotations

import json
import os

import httpx
from crewai import Agent, Crew, Process, Task
from crewai_tools import BaseTool


class RegulatoryFilingSearchTool(BaseTool):
    name: str = "Regulatory Filing Search"
    description: str = "Search SEC filing context and return cited regulatory evidence."

    def _run(self, query: str) -> str:
        response = httpx.post(
            os.getenv("RAG_API_URL", "http://rag_api:8000/query"),
            json={"question": query, "session_id": "crewai-risk", "top_k": 5},
            timeout=30,
        )
        response.raise_for_status()
        return json.dumps(response.json())


def build_crew(ticker: str) -> Crew:
    analyst = Agent(
        role="Senior Financial Analyst",
        goal=f"Evaluate {ticker} financial performance using verifiable evidence",
        backstory="A buy-side analyst who separates reported facts from assumptions.",
        verbose=True,
    )
    risk = Agent(
        role="Risk & Compliance Officer",
        goal=f"Identify material and regulatory risks for {ticker}",
        backstory="A fintech compliance specialist who requires filing citations.",
        tools=[RegulatoryFilingSearchTool()],
        verbose=True,
    )
    market = Agent(
        role="Market Intelligence Specialist",
        goal=f"Assess news, catalysts, and sentiment for {ticker}",
        backstory="A market researcher trained to flag stale or weak sources.",
        verbose=True,
    )
    writer = Agent(
        role="Executive Report Writer",
        goal="Create a concise balanced research brief without personalized advice",
        backstory="An editor who preserves uncertainty, citations, and reviewer comments.",
        verbose=True,
    )
    financial_task = Task(
        description=f"Analyze financial performance and ratios for {ticker}.",
        expected_output="A structured table with sources, periods, and caveats.",
        agent=analyst,
    )
    market_task = Task(
        description=f"Analyze current market intelligence and sentiment for {ticker}.",
        expected_output="Catalysts and sentiment with source dates and confidence.",
        agent=market,
    )
    risk_task = Task(
        description=f"Use the filing search tool to assess material risks for {ticker}.",
        expected_output="Prioritized risks with filing chunk citations.",
        agent=risk,
        context=[financial_task, market_task],
    )
    report_task = Task(
        description=f"Synthesize a balanced executive research brief for {ticker}.",
        expected_output="Markdown report with sections, citations, limitations, and no advice.",
        agent=writer,
        context=[financial_task, market_task, risk_task],
    )
    return Crew(
        agents=[analyst, market, risk, writer],
        tasks=[financial_task, market_task, risk_task, report_task],
        process=Process.sequential,
        memory=True,
        verbose=True,
    )
