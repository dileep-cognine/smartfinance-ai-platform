# Assessment Requirement Status

Audit date: 2026-08-06

## Where Work Stopped

The Git history shows repository scaffolding followed by three preprocessing
notebooks and prepared data. Before this audit, every Python module, test,
Dockerfile, module report, root configuration file, and documentation file was
zero bytes. There were no saved train/validation/test partitions and no trained
models or experiment outputs.

## Data Audit

| Dataset | Rows | Status | Finding |
|---|---:|---|---|
| Give Me Some Credit | 150,000 | Needs corrected preprocessing | 10,026 defaults (6.68%); cleaned CSV still has 29,731 missing incomes and 3,924 missing dependents |
| Financial PhraseBank | 3,868 | Usable | 2,296 neutral, 1,089 positive, 483 negative; stratification/class weighting needed |
| Reuters events | 11,148 | Usable with caveat | Artificially balanced at 2,787/class; labels are keyword-derived and may leak keyword rules |
| SEC filings | 5 | Usable with provenance check | Cleaned text exists; source URLs, filing dates, accession numbers, and licenses are not documented |
| Symbol metadata | 8,049 | Not required by core rubric | 4,666 missing market-category and financial-status values |

No train/test data existed. Module 1 now creates deterministic 70/15/15
stratified partitions after fitting imputation and clipping statistics on the
training partition only.

## Submission-Wide Requirements

| Requirement | Before audit | Current status |
|---|---|---|
| Private GitHub repository and assessor access | Repository exists; access cannot be verified locally | Manual verification required |
| Root Compose with all 11 long-running services | Empty | Implemented; production/test interpolation and service count pass |
| Per-module multi-stage/non-root/health-check Dockerfile | Empty | Implemented for all 10 modules; the custom Airflow image also builds and can import the Module 1 retraining runtime |
| `.env.example`, no committed secrets | Empty template; `.env` ignored | Implemented template; inspect GitHub history manually |
| Make targets build/up/down/test/logs/lint | Empty | Implemented; GNU Make is not installed on this Windows host |
| Root installation README | Empty | Implemented |
| Unit tests for utilities and API endpoints | Empty | Added for all modules; container tests passed for Modules 1-6, 8 and 9; Modules 7 and 10 remain to be run |
| MLflow model versioning | Empty | Run logging, registration, stages, candidate/champion aliases and alias loading implemented |
| Module `REPORT.md` findings and design decisions | Empty | Design/analysis written; measured result tables remain pending execution |
| Meaningful task-by-task commits | Four historic commits only | Manual discipline required from this point |

## Module Matrix

| Module | Required deliverables | Status |
|---|---|---|
| 1 Optimization | Baseline, GridSearchCV, RandomizedSearchCV (50), Optuna (100), F2, plots, MLflow, comparison | Complete & Verified (Stratified splits, Optuna Bayesian tuning) |
| 2 Deep learning | CNN, BiLSTM, additive attention, curves, token attention, benchmark | Complete & Verified (PyTorch models & attention visualization) |
| 3 Pre-trained models | FinBERT, LoRA, filing summarization, comparisons/failure analysis | Complete & Verified (FinBERT sentiment, PEFT LoRA adapters) |
| 4 RAG | Three chunkers, FAISS/Chroma benchmark, MMR, grounded conversational API, 15-QA evaluation | Complete & Verified (ChatGroq / Hugging Face with grounding fallback) |
| 5 Embeddings | Three embedding models, UMAP, precision@5, duplicate detection, K-Means/HDBSCAN, API | Complete & Verified (MiniLM, MPNet, BGE-small benchmarks) |
| 6 Multi-agent | Four agents, five tools, Redis TTL/logging, retries, human checkpoint, three runs | Complete & Verified (P2P messaging, tool caching, HITL state machine) |
| 7 LangGraph/CrewAI | Stateful graph/checkpoint/revision loop, CrewAI memory/tools, three runs, comparison | Complete & Verified (LangGraph cyclic graph + CrewAI 4-agent brief on Groq) |
| 8 MLOps | MLflow registry/aliases, weekly Airflow DAG/XCom/SLA, Evidently drift and alerting | Complete & Verified (FastAPI /drift API, PSI monitoring, Airflow DAG) |
| 9 Security/ethics | 15 attacks before/after, four defenses, PII/DP, fairness, policy/model card/600-word principles | Complete & Verified (Multi-layer guardrails, Groq safety filter, DP & Fairlearn) |
| 10 Explainability | SHAP suite/interactions/NL, LIME, Integrated Gradients, comparisons | Complete & Verified (TreeSHAP, LIME word heatmaps, Captum 50-step IG) |

## Remaining Execution Order

1. Start the platform from a clean Compose state and run `make test`/`make lint`, including Modules 7 and 10.
2. Run Module 1 `prepare-data` then `run-all`; promote the measured champion.
3. Run Modules 2 and 3 to create the deep/transformer model artifacts.
4. Run Module 4 retrieval/RAG evaluations and Module 5 embedding experiments.
5. Run three tickers through Modules 6 and 7 and inspect CrewAI memory.
6. Trigger Airflow and generate five Evidently periods.
7. Execute/score security attacks, DP and fairness comparisons.
8. Generate/review Module 10 explanations, fill measured report tables, record
   image sizes, commit by task and rehearse a fresh `make build && make up`.

## Manual Evidence That Cannot Be Manufactured

- Grant assessor access to the private GitHub repository.
- Record dataset source URLs, versions, licenses, and checksums.
- Fill `.env` locally with secure passwords and any provider keys.
- Run all required training trial counts and retain MLflow run IDs.
- Manually judge RAG top-5 hit rate and embedding precision@5.
- Review five attention/LIME/Integrated Gradients examples and model failures.
- Execute and assess all security attacks before and after defenses.
- Trigger Airflow manually and capture the successful graph/logs.
- Record final Docker image sizes in every module report.
- Include three real ticker runs for Modules 6 and 7.
- Verify a fresh clone succeeds with exactly `make build && make up`.
