# Module 5: Embeddings and Document Similarity

This FastAPI service ranks Financial PhraseBank sentences against supplied
text. The running API uses TF-IDF unigram/bigram vectors and cosine similarity,
and reports its model as `tfidf-baseline`. It does not require provider keys
or download embedding-model weights for these API requests.

## Required data and configuration

Run the setup and Docker commands below from the **repository root**.
Docker Desktop must be running in Linux container mode.

- Keep your existing root `.env`. If missing, copy `.env.example` to `.env`.
- Provide `data/processed/cleaned/financial_phrasebank_clean.csv` with a
  `Sentence` column containing nonempty text. Other columns are ignored by
  this API.
- `DATA_DIR` points to the parent data directory. Compose uses `/app/data`.

## Run with Docker

```powershell
docker compose up -d --build embeddings_service
docker compose ps
docker compose logs --tail=100 embeddings_service
```

Open [interactive API docs](http://localhost:8002/docs). This service has no
Compose service dependencies.

## Run locally with Python

Use Python 3.11, matching the container runtime. From the repository root,
create a virtual environment if needed and install the shared dependencies:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The shared requirements install dependencies for all modules. Use Docker if
native dependencies cannot be installed on Windows. Then start the service:

```powershell
cd module_05_embeddings
$env:DATA_DIR = (Resolve-Path ..\data).Path
..\venv\Scripts\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8002
```

Local execution does not automatically load the root `.env`. Stop the server
with Ctrl+C.

## Try a request

In a second PowerShell terminal (for either execution method):

```powershell
Invoke-RestMethod -Uri 'http://localhost:8002/health'

$body = @{
    document_text = 'The company reported higher revenue and operating profit.'
    top_k = 3
} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://localhost:8002/similar-docs' -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6
```

The health response includes `status` and `documents`. Search returns `model`
and `results`, each containing `document_id`, `text`, and `score`. IDs have the
form `phrasebank-0`. Results are ordered by descending similarity and can
include zero scores; scores are not calibrated confidence probabilities.
`document_text` accepts 2–20,000 characters and `top_k` accepts 1–25.

## Tests

From the repository root, using the built Docker image:

```powershell
docker compose run --rm --no-deps embeddings_service python -m pytest tests -q
```

Or from this module directory with dependencies installed:

```powershell
..\venv\Scripts\python.exe -m pytest tests -q
```

## Outputs and troubleshooting

- The API returns JSON and keeps its index in memory; it does not save a
  persistent vector database or report files. Restart after changing the CSV.
- A missing CSV creates a one-document placeholder corpus. A healthy response
  alone does not prove the intended dataset was loaded.
- An empty CSV or missing `Sentence` values can prevent indexing. Supply
  nonempty sentences with meaningful words.
- For HTTP 422, check field names and limits against `/docs`.
- If port 8002 is occupied, stop the existing service or choose another local
  Uvicorn port and update the request URLs.
- Stop the Docker service with `docker compose stop embeddings_service`.

See [REPORT.md](REPORT.md) for assessment context and the
[platform README](../README.md) for the complete stack.
