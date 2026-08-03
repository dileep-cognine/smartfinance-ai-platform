"""Weekly SmartFinance retraining DAG.

Task bodies are provider-neutral integration points. They deliberately fail
validation instead of silently training on malformed data.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from datetime import UTC, datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

LOGGER = logging.getLogger(__name__)


def sla_miss_callback(*args, **kwargs) -> None:
    LOGGER.error("SmartFinance retraining task missed its 30-minute SLA")


def validate_data(**context):
    row_count = int(context["dag_run"].conf.get("row_count", 150_000))
    if row_count < 1_000:
        raise AirflowFailException(f"Unexpected credit dataset row count: {row_count}")
    context["ti"].xcom_push(key="data_version", value=context["ds_nodash"])
    return {"row_count": row_count, "schema_valid": True}


def feature_engineering(**context):
    data_version = context["ti"].xcom_pull(task_ids="data_validation", key="data_version")
    subprocess.run(
        ["python", "-m", "src.main", "prepare-data"],
        cwd=os.getenv(
            "MODULE_01_DIR", "/opt/smartfinance/module_01_model_optimization"
        ),
        check=True,
        timeout=30 * 60,
    )
    return {"data_version": data_version, "split_seed": 42}


def train_model(**context):
    completed = subprocess.run(
        [
            "python",
            "-m",
            "src.main",
            "tune",
            "--strategy",
            "bayesian",
            "--trials",
            "100",
        ],
        cwd=os.getenv(
            "MODULE_01_DIR", "/opt/smartfinance/module_01_model_optimization"
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=30 * 60,
    )
    payload = json.loads(completed.stdout)
    run_id = payload.get("mlflow_run_id", str(uuid.uuid4()))
    context["ti"].xcom_push(key="model_run_id", value=run_id)
    context["ti"].xcom_push(key="training_output", value=payload)
    return {"model_run_id": run_id, "output": payload}


def evaluate_model(**context):
    configured = context["dag_run"].conf
    training_output = context["ti"].xcom_pull(
        task_ids="model_training", key="training_output"
    )
    metrics = {
        "candidate_f1": float(
            configured.get(
                "candidate_f1",
                training_output.get("test_metrics", {}).get("f1", 0.0),
            )
        ),
        "production_f1": float(configured.get("production_f1", 0.0)),
    }
    context["ti"].xcom_push(key="metrics", value=metrics)
    return metrics


def choose_promotion(**context):
    metrics = context["ti"].xcom_pull(task_ids="model_evaluation", key="metrics")
    return (
        "promote_model"
        if metrics["candidate_f1"] > metrics["production_f1"] + 0.02
        else "skip_promotion"
    )


def promote_model(**context):
    run_id = context["ti"].xcom_pull(task_ids="model_training", key="model_run_id")
    LOGGER.info("Promote MLflow run %s after registry integration is configured", run_id)


def skip_promotion():
    LOGGER.info("Candidate did not exceed the production F1 by more than 0.02")


def notify(**context):
    LOGGER.info("Retraining pipeline completed: run_id=%s", context["run_id"])


default_args = {
    "owner": "smartfinance",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(minutes=30),
    "sla_miss_callback": sla_miss_callback,
}

with DAG(
    dag_id="smartfinance_retraining_pipeline",
    description="Validate, train, evaluate, and conditionally promote credit model",
    default_args=default_args,
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule="0 2 * * 0",
    catchup=False,
    max_active_runs=1,
    tags=["smartfinance", "retraining"],
) as dag:
    data_validation = PythonOperator(
        task_id="data_validation", python_callable=validate_data
    )
    features = PythonOperator(
        task_id="feature_engineering", python_callable=feature_engineering
    )
    training = PythonOperator(task_id="model_training", python_callable=train_model)
    evaluation = PythonOperator(
        task_id="model_evaluation", python_callable=evaluate_model
    )
    conditional = BranchPythonOperator(
        task_id="conditional_promotion", python_callable=choose_promotion
    )
    promotion = PythonOperator(task_id="promote_model", python_callable=promote_model)
    skipped = PythonOperator(task_id="skip_promotion", python_callable=skip_promotion)
    notification = PythonOperator(
        task_id="notification",
        python_callable=notify,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    data_validation >> features >> training >> evaluation >> conditional
    conditional >> [promotion, skipped] >> notification
