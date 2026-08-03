# Module 8 Report - MLOps

The platform defines PostgreSQL-backed MLflow, Airflow webserver/scheduler,
named artifact/log volumes, and a weekly Sunday 02:00 UTC retraining DAG. The
DAG validates row count, invokes Module 1 preprocessing and 100-trial Bayesian
tuning, passes data version/run ID/metrics through XCom, enforces 30-minute
SLAs, and promotes only above an absolute 0.02 improvement.

The registry adapter creates the model, tags versions with team/version/data
version/training date, transitions Staging then Production, assigns `candidate`
and `champion` aliases, and loads production strictly through the champion
alias.

Monitoring covers data drift through Evidently and per-feature PSI. The
simulation adds progressively stronger location/noise shifts and writes five
HTML/JSON periods. PSI above 0.2 is the retraining threshold: below 0.1 is
normally stable, 0.1-0.2 warrants investigation, and above 0.2 triggers the
Airflow REST endpoint. Prediction probability drift and labelled accuracy/F1
must be added to the production batch schema when those fields arrive.

Operational evidence still requires starting the platform, manually triggering
the DAG, confirming every task, demonstrating alias replacement, and retaining
MLflow/Airflow screenshots and five generated reports. Slack/email credentials
must be configured for real delivery rather than log-only notification.

Base image: `python:3.11.9-slim-bookworm`. Record measured image size.
