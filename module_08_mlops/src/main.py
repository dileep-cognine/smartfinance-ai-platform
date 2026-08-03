"""Model monitoring API with PSI-triggered retraining alerts."""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .monitoring import population_stability_index

app = FastAPI(title="SmartFinance Monitoring API", version="0.1.0")


class DriftRequest(BaseModel):
    reference: dict[str, list[float]]
    current: dict[str, list[float]]
    threshold: float = Field(default=0.2, gt=0)
    trigger_retraining: bool = False


class DriftResponse(BaseModel):
    psi: dict[str, float]
    drifted_features: list[str]
    retraining_triggered: bool


def _trigger_airflow() -> bool:
    base_url = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080")
    try:
        response = httpx.post(
            f"{base_url}/api/v1/dags/smartfinance_retraining_pipeline/dagRuns",
            json={"conf": {"trigger": "psi_alert"}},
            auth=(
                os.getenv("AIRFLOW_ADMIN_USER", "admin"),
                os.getenv("AIRFLOW_ADMIN_PASSWORD", "admin"),
            ),
            timeout=10,
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/drift", response_model=DriftResponse)
def drift(request: DriftRequest) -> DriftResponse:
    if set(request.reference) != set(request.current):
        raise HTTPException(status_code=422, detail="Reference/current features must match")
    try:
        values = {
            feature: population_stability_index(
                request.reference[feature], request.current[feature]
            )
            for feature in request.reference
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    drifted = [feature for feature, value in values.items() if value > request.threshold]
    triggered = bool(drifted and request.trigger_retraining and _trigger_airflow())
    return DriftResponse(
        psi=values,
        drifted_features=drifted,
        retraining_triggered=triggered,
    )
