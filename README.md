# SmartFinance AI Platform

Backend-only AI/ML assessment platform for credit-risk optimization, financial
text modelling, retrieval-augmented generation, agent orchestration, MLOps,
security, fairness, and explainability.

> Current implementation status is tracked in
> [docs/ASSESSMENT_STATUS.md](docs/ASSESSMENT_STATUS.md). A directory existing
> does not mean its assessment tasks are complete.

## Architecture

The root Compose project starts PostgreSQL, Redis, MLflow, Airflow webserver and
scheduler, five backend APIs, and Jupyter Lab. Stateful services use named
volumes. Application containers mount the repository data read-only.

| Service | URL | Purpose |
|---|---|---|
| MLflow | http://localhost:5000 | Experiments and model registry |
| Airflow | http://localhost:8080 | Weekly retraining DAG (`admin` / configured password) |
| RAG API | http://localhost:8001/docs | Filing Q&A |
| Embeddings API | http://localhost:8002/docs | Semantic search and duplicates |
| Agent API | http://localhost:8003/docs | Research orchestration |
| Monitoring API | http://localhost:8004/docs | Drift and performance monitoring |
| Security/XAI API | http://localhost:8005/docs | Guardrails, privacy, fairness, explanations |
| Jupyter Lab | http://localhost:8888 | Exploration only |

Ports are bound to `127.0.0.1`; they are not exposed on the LAN.

## Module execution guides

Each module has its own README with execution instructions. Run Docker Compose
commands from the repository root; run `python -m src.main` commands from the
specific module directory as directed by its guide.

| Module | Guide |
|---|---|
| 1 - Credit-risk optimization | [README](module_01_model_optimization/README.md) |
| 2 - Deep learning | [README](module_02_deep_learning/README.md) |
| 3 - Pre-trained models | [README](module_03_pretrained_models/README.md) |
| 4 - RAG pipeline | [README](module_04_rag_pipeline/README.md) |
| 5 - Embeddings | [README](module_05_embeddings/README.md) |
| 6 - Multi-agent research | [README](module_06_multi_agent/README.md) |
| 7 - LangGraph and CrewAI | [README](module_07_langgraph_crewai/README.md) |
| 8 - MLOps | [README](module_08_mlops/README.md) |
| 9 - Security and ethics | [README](module_09_security_ethics/README.md) |
| 10 - Explainability | [README](module_10_explainability/README.md) |

## Prerequisites

- Docker Engine 24.0 or newer
- Docker Compose V2.20 or newer
- Optional: GNU Make 4.0 or newer for the Make targets
- Git 2.40 or newer
- Optional: NVIDIA Container Toolkit for GPU training modules

On Windows, run the Make targets from Git Bash, WSL, or install GNU Make.

## Docker quick start (PowerShell)

Open Docker Desktop in Linux container mode and wait for it to start. Open a
terminal in the repository root (the folder containing `docker-compose.yml`).
These commands do not require Make or a local Python environment.

### 1. Configure the environment

Create `.env` only if it does not already exist:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Review `.env` before starting. Keep the database password consistent with the
Airflow database connection string, and configure the Airflow secret and admin
password. Provider API keys can remain blank for supported local fallback paths.

### 2. Build and start the platform

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose ps
```

The first build downloads dependencies and can take several minutes. Wait for
services to become healthy, then open the URLs in the architecture table above.
Use `/docs` to try the backend APIs; there is no frontend application.
Airflow uses the admin credentials in `.env`; Jupyter uses `JUPYTER_TOKEN`.

### 3. Check health and logs

```powershell
Invoke-RestMethod -Uri 'http://localhost:8001/health'
Invoke-RestMethod -Uri 'http://localhost:8002/health'
docker compose logs --tail=100
```

To follow a specific service's logs (Ctrl+C exits log viewing):

```powershell
docker compose logs -f rag_api
```

### Run individual services

For example, start only the embeddings service or the RAG service and its
configured dependencies:

```powershell
docker compose up -d --build embeddings_service
docker compose up -d --build rag_api
```

Other API service names are `agent_orchestrator`, `monitoring`, and
`security_xai`. See the module guides for requests and required datasets.

### Run training commands

Training-profile services do not start with the default stack. Run them as
one-off jobs. For example, prepare Module 1 data before tuning:

```powershell
docker compose --profile training build model_optimizer
docker compose --profile training run --rm model_optimizer python -m src.main prepare-data
docker compose --profile training run --rm model_optimizer python -m src.main tune --strategy bayesian --trials 100
```

### Stop and restart

Stop containers while keeping them available to restart:

```powershell
docker compose stop
docker compose start
```

Or remove containers and the project network while retaining named data volumes:

```powershell
docker compose down
```

Run `docker compose up -d` to start again after `down`. Avoid adding `--volumes`
unless you intend to delete the stored service data.

## Installation

### Step 1 - Clone the Repository

```bash
git clone https://github.com/dileep-cognine/smartfinance-ai-platform.git
cd smartfinance-ai-platform
```

### Step 2 - Configure Environment Variables

```bash
cp .env.example .env
```

Replace database passwords and the Airflow secret. Provider keys may remain
blank for code paths that support local deterministic fallbacks. Never commit
`.env`.

### Step 3 - Build All Docker Images

```bash
make build
```

The first build downloads model/runtime layers and may take 10-20 minutes.

### Step 4 - Start All Services

```bash
make up
make ps
```

Wait until dependency services report `healthy`. Inspect failures with
`docker compose logs <service>`.

### Step 5 - Verify Service Endpoints

Open the URLs in the architecture table. API health checks are also available
at `/health`.

### Step 6 - Run Tests

```bash
make test
make lint
```

### Step 7 - Run a Module Script

Module 1 builds deterministic train/validation/test partitions before tuning:

```bash
docker compose --profile training run --rm model_optimizer \
  python -m src.main prepare-data

docker compose --profile training run --rm model_optimizer \
  python -m src.main tune --strategy bayesian --trials 100
```

## Data

- Credit risk: Kaggle *Give Me Some Credit*, 150,000 labelled rows.
- Sentiment: Financial PhraseBank, 3,868 labelled sentences.
- Events: Reuters-derived balanced set, 11,148 sentences.
- RAG: five locally stored SEC 10-K text documents.

Raw data provenance and licensing must be confirmed before submission. Generated
splits and artifacts are reproducible and should not be committed.

## Standard Commands

```bash
make build    # Rebuild without cache
make up       # Start the platform
make down     # Stop and remove project volumes
make logs     # Follow logs
make test     # Test backend services inside containers
make lint     # Run Ruff inside containers
make ps       # Show status
make config   # Validate Compose interpolation
```

`make clean` performs a host-wide Docker prune and must be used with caution.

## Design Decisions

- APIs are backend-only FastAPI services; no frontend is included.
- Secrets are injected through `.env`, never image layers.
- Runtime images use pinned bases, multi-stage builds, non-root users, and
  health checks.
- Training data is split with stratification and a fixed seed to make metrics
  comparable and prevent the imbalanced default target from drifting between
  partitions.
- MLflow and Airflow use separate PostgreSQL databases created on first startup.

## Known Limitations

The repository began as an empty scaffold plus exploratory notebooks. Consult
the status matrix before presenting it as complete. Model training outputs,
manual retrieval judgements, red-team transcripts, image sizes, screenshots,
and report findings must be generated and reviewed by the candidate.
