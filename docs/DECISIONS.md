# Architecture Decisions

1. PostgreSQL is retained because the assessment explicitly requires a shared
   production-capable metadata database for MLflow and Airflow.
2. Module dependencies remain isolated in per-image requirements files.
3. Training jobs use Compose profiles so the default platform remains the
   required 11 long-running services.
4. Provider-free deterministic fallbacks keep health checks available, while
   assessment experiments explicitly record the configured dense/LLM model.
5. Measured results and manual judgements are never fabricated in reports.
