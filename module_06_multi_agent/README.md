# Module 6: Multi-Agent Research Orchestrator

This module implements a human-in-the-loop, multi-agent financial research platform with structured JSON messaging, tool caching, exponential backoff retries, and REST API orchestration.

---

## Architecture & Workflow

```mermaid
flowchart LR
  User([User / Client]) -->|POST /research/start| Orchestrator[Agent Orchestrator]
  
  subgraph Parallel Phase [Parallel Execution]
    Orchestrator -->|Worker 1| DataFetcher[DataFetcherAgent]
    Orchestrator -->|Worker 2| NewsAnalysis[NewsAnalysisAgent]
  end
  
  Parallel Phase --> RiskAssessment[RiskAssessmentAgent]
  RiskAssessment --> Checkpoint{Human Checkpoint<br/>awaiting_confirmation}
  
  Checkpoint -->|POST /research/{id}/confirm<br/>approved: true| ReportWriter[ReportWriterAgent]
  Checkpoint -->|approved: false| Stop([Failed / Cancelled])
  
  ReportWriter --> FinalReport([Investment Brief Report])
  
  DataFetcher -.->|Cache / Get| Cache[(Redis / Memory TTL Cache)]
  NewsAnalysis -.->|Cache / Get| Cache
  RiskAssessment -.->|Query| RAG[(Module 4 RAG API)]
```

### Agent Roles
- **`DataFetcherAgent`**: Retrieves daily stock price series and company financial overview ratios.
- **`NewsAnalysisAgent`**: Analyzes market sentiment across recent news articles.
- **`RiskAssessmentAgent`**: Evaluates regulatory and compliance risks by querying regulatory filings.
- **`ReportWriterAgent`**: Synthesizes market data, sentiment scores, and risk findings into a structured Investment Research Brief.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status check |
| `POST` | `/research/start` | Initiates research on a ticker (e.g. `{"ticker": "AAPL"}`) and halts at `awaiting_confirmation` |
| `POST` | `/research/{run_id}/confirm` | Approves (`{"approved": true}`) or rejects (`{"approved": false}`) research to finalize report |
| `GET` | `/docs` | Interactive OpenAPI / Swagger UI |

---

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL for tool response caching (falls back to in-memory TTL) |
| `RAG_API_URL` | `http://rag_api:8000/query` | Module 4 RAG API endpoint for regulatory search |
| `ALPHA_VANTAGE_API_KEY` | `""` | Alpha Vantage API key (falls back to deterministic simulation when unconfigured) |
| `TOOL_CACHE_TTL_SECONDS` | `3600` | Tool result cache expiration time in seconds (1 hour) |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Timeout threshold for external tool calls |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## Running Locally

### 1. Start the FastAPI Service

From the `module_06_multi_agent` directory:

```powershell
$env:PYTHONPATH = "."
python -m uvicorn src.main:app --host 127.0.0.1 --port 8003 --reload
```

Interactive API documentation will be accessible at [http://127.0.0.1:8003/docs](http://127.0.0.1:8003/docs).

---

## Testing & Verification

### 1. Run Automated Unit Tests

```powershell
$env:PYTHONPATH = "."
python -m pytest tests -v
```

### 2. Code Linting & Style Checks

```powershell
python -m ruff check .
```

### 3. End-to-End API Test Flow

#### Step A: Initiate Research
```powershell
$resp = Invoke-RestMethod -Uri "http://127.0.0.1:8003/research/start" -Method Post -ContentType "application/json" -Body '{"ticker": "AAPL"}'
$resp | ConvertTo-Json -Depth 5
```

#### Step B: Confirm & Generate Report
```powershell
$result = Invoke-RestMethod -Uri "http://127.0.0.1:8003/research/$($resp.run_id)/confirm" -Method Post -ContentType "application/json" -Body '{"approved": true}'
Write-Output $result.final_report
```

---

## Docker Support

### Build & Run Containerized Tests

From the repository root:

```powershell
docker build -t module_06_agent -f module_06_multi_agent/Dockerfile .
docker run --rm module_06_agent pytest -v
```

### Docker Compose Integration

The service is configured as `agent_orchestrator` in `docker-compose.yml` and is mapped to port `8003`.

```powershell
docker compose up -d agent_orchestrator
```
