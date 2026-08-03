# System Design

Data preparation produces versioned deterministic splits. Training modules log
metrics and artifacts to MLflow. The RAG service indexes SEC filings and feeds
risk agents. Agent workflows pause before final report generation. Airflow
retraining validates data, trains, evaluates and conditionally promotes the
MLflow champion. Monitoring can trigger retraining but cannot approve a model.
Security, privacy, fairness and explainability controls operate across these
boundaries.
