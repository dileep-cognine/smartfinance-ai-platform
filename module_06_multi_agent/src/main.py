"""Human-in-the-loop research orchestration API."""

from __future__ import annotations

import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .communication.messaging import ResearchState
from .orchestration.orchestrator import Orchestrator

app = FastAPI(title="SmartFinance Agent Orchestrator", version="0.1.0")
orchestrator = Orchestrator()
states: dict[str, ResearchState] = {}


class StartRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)


class ConfirmRequest(BaseModel):
    approved: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/research/start", response_model=ResearchState)
def start_research(request: StartRequest) -> ResearchState:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,9}", request.ticker):
        raise HTTPException(status_code=422, detail="Invalid ticker format")
    try:
        state = orchestrator.start(request.ticker)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    states[state.run_id] = state
    return state


@app.post("/research/{run_id}/confirm", response_model=ResearchState)
def confirm_research(run_id: str, request: ConfirmRequest) -> ResearchState:
    state = states.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Research run not found")
    if state.status != "awaiting_confirmation":
        raise HTTPException(status_code=409, detail="Run is not awaiting confirmation")
    if not request.approved:
        state.status = "failed"
        return state
    return orchestrator.finalize(state)
