"""Small MLflow adapter that keeps training code testable."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow


@contextmanager
def tracked_run(
    run_name: str, enabled: bool = True, tags: dict[str, str] | None = None
) -> Iterator[Any | None]:
    """Start an MLflow run when tracking is enabled."""
    if not enabled:
        yield None
        return
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("smartfinance-credit-risk")
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        yield run


def log_run(
    parameters: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: list[Path],
    model: Any | None = None,
) -> None:
    """Log sanitized parameters, metrics, artifacts, and an optional model."""
    mlflow.log_params({key: str(value) for key, value in parameters.items()})
    mlflow.log_metrics(metrics)
    for path in artifact_paths:
        if path.exists():
            mlflow.log_artifact(str(path))
    if model is not None:
        mlflow.sklearn.log_model(model, artifact_path="model")
