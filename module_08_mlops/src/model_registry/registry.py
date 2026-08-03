"""MLflow credit-model registration, promotion, aliases, and loading."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "smartfinance-credit-risk"


def register_run(
    run_id: str,
    data_version: str,
    version_tag: str,
    client: MlflowClient | None = None,
) -> str:
    client = client or MlflowClient()
    try:
        client.get_registered_model(MODEL_NAME)
    except MlflowException:
        client.create_registered_model(
            MODEL_NAME,
            tags={"team": "smartfinance", "intended_use": "credit-risk"},
        )
    version = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
        tags={
            "team": "smartfinance",
            "version": version_tag,
            "data_version": data_version,
            "training_date": datetime.now(UTC).date().isoformat(),
        },
    )
    client.set_registered_model_alias(MODEL_NAME, "candidate", version.version)
    return str(version.version)


def promote_version(version: str, client: MlflowClient | None = None) -> None:
    client = client or MlflowClient()
    client.transition_model_version_stage(MODEL_NAME, version, "Staging")
    client.transition_model_version_stage(
        MODEL_NAME, version, "Production", archive_existing_versions=True
    )
    client.set_registered_model_alias(MODEL_NAME, "champion", version)


def load_production_model() -> Any:
    """Load the alias, never a hard-coded artifact or version."""
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")


def promote_if_improved(
    candidate_version: str,
    candidate_f1: float,
    production_f1: float,
    minimum_improvement: float = 0.02,
    client: MlflowClient | None = None,
) -> bool:
    if candidate_f1 <= production_f1 + minimum_improvement:
        return False
    promote_version(candidate_version, client)
    return True
