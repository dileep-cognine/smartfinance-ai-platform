# Module 4: Retrieval-Augmented Generation (RAG)

This FastAPI service answers questions using local SEC filing text, returns
source citations, and keeps conversation history by session. The default
retriever uses an in-memory TF-IDF index. Without provider keys, answers use
matching sentences from retrieved documents rather than an external LLM.

## Required data and configuration

Run the setup and Docker commands below from the **repository root**.
Docker Desktop must be running in Linux container mode.

- Keep your existing root `.env`. If it is missing, copy `.env.example` to `.env`.
- Put cleaned filings in `data/processed/cleaned/sec_filings/`, named
  `*_cleaned.txt` (UTF-8 text).
- `DATA_DIR` is the parent data directory; Compose sets it to `/app/data`.
- `GROQ_API_KEY` with `GROQ_MODEL` optionally enables Groq generation.
  `HUGGINGFACE_TOKEN` with `HF_MODEL` optionally enables Hugging Face generation
  when Groq is not configured. Keys can remain blank for the local baseline.
- `RAG_MEMORY_TURNS` controls retained turns per session (default: `5`).

## Run with Docker

```powershell
docker compose up -d --build rag_api
docker compose ps
docker compose logs --tail=100 rag_api
```

Compose also starts the configured MLflow dependency. Open
[interactive API docs](http://localhost:8001/docs).

## Run locally with Python

Use Python 3.11, matching the container runtime. From the repository root,
create a virtual environment if needed and install the RAG dependencies:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r module_04_rag_pipeline/requirements.txt
```

These dependencies include the RAG API, provider integrations, and retrieval
benchmarks, without CrewAI or unrelated training and explainability packages.
Docker is an alternative if native dependencies fail to install on Windows.
Then start this module:

```powershell
cd module_04_rag_pipeline
$env:DATA_DIR = (Resolve-Path ..\data).Path
..\venv\Scripts\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8001
```

Local execution does not automatically load the root `.env`. Set optional
provider variables in your terminal when needed. Stop the server with Ctrl+C.

## Try a request

In a second PowerShell terminal (for either execution method):

```powershell
Invoke-RestMethod -Uri 'http://localhost:8001/health'

$body = @{
    question = 'What risks are described in the filings?'
    session_id = 'readme-demo'
    top_k = 3
} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://localhost:8001/query' -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6

Invoke-RestMethod -Uri 'http://localhost:8001/sessions/readme-demo' | ConvertTo-Json -Depth 6
```

`/query` returns `answer`, `citations`, and `session_id`. Each citation contains
the source filename, chunk number, score, and text. Unsupported questions may
return `Insufficient information`. `top_k` accepts 1–10; session IDs accept
letters, digits, underscores, and hyphens (1–128 characters).

## Tests

From the repository root, using the built Docker image:

```powershell
docker compose -f docker-compose.yml -f docker/compose/docker-compose.test.yml run --build --rm --no-deps rag_api python -m pytest tests -q
```

Or from this module directory with dependencies installed:

```powershell
..\venv\Scripts\python.exe -m pytest tests -q
```

## Outputs and troubleshooting

- Responses are returned as JSON; this API does not write report files.
- The index and session history live in memory and are lost when the process
  restarts. Restart after changing corpus files to rebuild the index.
- A healthy response does not confirm that filings exist: missing files create
  a placeholder corpus. Check the data path and filename pattern if answers
  contain no useful citations.
- For HTTP 422, check the request fields against `/docs`.
- If port 8001 is occupied, stop the existing service or choose another local
  Uvicorn port and update the request URLs.
- Stop the Docker service with `docker compose stop rag_api` from the root.

See [REPORT.md](REPORT.md) for assessment context and the
[platform README](../README.md) for the complete stack.
