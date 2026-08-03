# API Reference

OpenAPI specifications are served at `/docs` on ports 8001 through 8005.

- `POST /query`: grounded filing question with `question` and `session_id`.
- `POST /similar-docs`: semantic document lookup.
- `POST /research/start` and `POST /research/{run_id}/confirm`: agent workflow.
- `POST /drift`: PSI drift calculation and optional retraining trigger.
- `POST /guard`, `/anonymize`, and `/explain`: security/privacy/XAI controls.
