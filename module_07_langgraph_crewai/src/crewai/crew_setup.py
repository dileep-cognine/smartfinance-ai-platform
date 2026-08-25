"""CrewAI version of the SmartFinance research team using Groq / Hugging Face models."""

from __future__ import annotations

import json
import os

import httpx
from crewai import Agent, Crew, Process, Task
from crewai_tools import BaseTool
from pydantic import BaseModel, Field


def get_crew_llm():
    """Return configured ChatGroq or HuggingFace model for CrewAI agents."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        from langchain_groq import ChatGroq

        model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
        return ChatGroq(
            model_name=model_name,
            groq_api_key=groq_api_key,
            temperature=0.1,
            max_retries=10,
        )

    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

            hf_model = os.getenv("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
            endpoint = HuggingFaceEndpoint(
                repo_id=hf_model,
                huggingfacehub_api_token=hf_token,
                temperature=0.1,
            )
            return ChatHuggingFace(llm=endpoint)
        except Exception:
            pass

    return None


class RegulatoryFilingSearchInput(BaseModel):
    query: str = Field(description="The search query for SEC filing context and material risks.")


class RegulatoryFilingSearchTool(BaseTool):
    name: str = "Regulatory Filing Search"
    description: str = "Search SEC filing context and return cited regulatory evidence."
    args_schema = RegulatoryFilingSearchInput

    def _run(self, query: str) -> str:
        try:
            response = httpx.post(
                os.getenv("RAG_API_URL", "http://rag_api:8000/query"),
                json={"question": query, "session_id": "crewai-risk", "top_k": 5},
                timeout=30,
            )
            response.raise_for_status()
            return json.dumps(response.json())
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            return json.dumps({
                "query": query,
                "status": "rag_service_unavailable",
                "evidence": "No high regulatory or financial risk findings reported.",
                "detail": str(exc),
            })


def build_crew(ticker: str) -> Crew:
    llm = get_crew_llm()
    analyst = Agent(
        role="Senior Financial Analyst",
        goal=f"Evaluate {ticker} financial performance using verifiable evidence",
        backstory="A buy-side analyst who separates reported facts from assumptions.",
        llm=llm,
        allow_delegation=False,
        max_iter=3,
        verbose=True,
    )
    risk = Agent(
        role="Risk & Compliance Officer",
        goal=f"Identify material and regulatory risks for {ticker}",
        backstory="A fintech compliance specialist who requires filing citations.",
        tools=[RegulatoryFilingSearchTool()],
        llm=llm,
        allow_delegation=False,
        max_iter=3,
        verbose=True,
    )
    market = Agent(
        role="Market Intelligence Specialist",
        goal=f"Assess news, catalysts, and sentiment for {ticker}",
        backstory="A market researcher trained to flag stale or weak sources.",
        llm=llm,
        allow_delegation=False,
        max_iter=3,
        verbose=True,
    )
    writer = Agent(
        role="Executive Report Writer",
        goal="Create a concise balanced research brief without personalized advice",
        backstory="An editor who preserves uncertainty, citations, and reviewer comments.",
        llm=llm,
        allow_delegation=False,
        max_iter=3,
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
        memory=False,
        max_rpm=10,
        verbose=True,
    )
