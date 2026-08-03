# Architecture

The default Compose platform contains 11 services: PostgreSQL, Redis, MLflow,
Airflow webserver and scheduler, RAG, embeddings, agent orchestration,
monitoring, security/XAI, and Jupyter. Training-only module images are exposed
through the `training` profile. PostgreSQL owns isolated MLflow and Airflow
databases; Redis caches agent tools; application services communicate on the
private `smartfinance` network.
