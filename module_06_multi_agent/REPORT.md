# Module 6 Report - Multi-Agent Research Orchestrator

## 1. System Architecture

The multi-agent research platform coordinates four specialized agents with typed schemas, in-memory/Redis TTL caching, exponential backoff retries, and an explicit human confirmation checkpoint.

```mermaid
flowchart LR
  User([User / API Client]) -->|POST /research/start| Orchestrator[Agent Orchestrator]
  
  subgraph Concurrent Phase [Parallel Execution]
    Orchestrator -->|ThreadPool Worker 1| DataFetcher[DataFetcherAgent]
    Orchestrator -->|ThreadPool Worker 2| NewsAnalysis[NewsAnalysisAgent]
  end
  
  Concurrent Phase --> RiskAssessment[RiskAssessmentAgent]
  RiskAssessment --> Checkpoint{Human Checkpoint<br/>awaiting_confirmation}
  
  Checkpoint -->|POST /research/{id}/confirm<br/>approved: true| ReportWriter[ReportWriterAgent]
  Checkpoint -->|approved: false| Stop([Failed / Cancelled])
  
  ReportWriter --> FinalReport([Investment Research Brief])
  
  DataFetcher -.->|Cache Read/Write| Cache[(Redis / Memory TTL)]
  NewsAnalysis -.->|Cache Read/Write| Cache
  RiskAssessment -.->|Query| RAG[(Module 4 RAG API)]
```

---

## 2. Benchmark Timing & Metrics

The workflow was executed across three representative tickers (**AAPL**, **MSFT**, and **TSLA**), recording execution latencies per agent and live financial metrics.

### Agent Execution Latency (ms)

| Ticker | DataFetcherAgent | NewsAnalysisAgent | RiskAssessmentAgent | ReportWriterAgent | Total Pre-Confirmation Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | 4,605.72 ms | 2,290.85 ms | 4,982.79 ms | 0.11 ms | **~9,588 ms** |
| **MSFT** | 4,577.09 ms | 2,294.12 ms | 4,964.16 ms | 0.18 ms | **~9,541 ms** |
| **TSLA** | 4,581.54 ms | 2,283.04 ms | 5,041.77 ms | 0.30 ms | **~9,623 ms** |

### Live Provider Data Verification (AAPL)

- **Stock Price Time Series**: Successfully retrieved compact daily OHLCV series.
- **Financial Ratios**:
  - **P/E Ratio**: `35.48`
  - **PEG Ratio**: `2.494`
  - **Price-to-Book Ratio**: `41.99`
  - **Profit Margin**: `27.6%`
- **News Sentiment Score**: `+0.1625` (computed across 50 live news articles).
- **Regulatory Risk**: Assessed via query `What are the material regulatory and financial risks for AAPL?`.

---

## 3. Bottleneck Analysis & Parallelization Plan

### Identified Bottlenecks
1. **RiskAssessmentAgent Latency (~5,000 ms)**: External network query for filing search with DNS resolution / fallback handling constitutes ~52% of total pipeline latency.
2. **DataFetcherAgent Internal Serialization (~4,600 ms)**: Inside `DataFetcherAgent`, `get_stock_data` and `calculate_financial_ratios` are currently called sequentially instead of concurrently.

### Parallel Optimization Strategy
- **Current Execution**:
  $$\text{Total Time} \approx \max(T_{\text{DataFetcher}}, T_{\text{NewsAnalysis}}) + T_{\text{RiskAssessment}} \approx 4.6\text{s} + 5.0\text{s} \approx 9.6\text{s}$$
- **Full Parallel Plan**:
  By dispatching `DataFetcherAgent`, `NewsAnalysisAgent`, and `RiskAssessmentAgent` concurrently in a single 3-worker pool before the confirmation barrier, total execution time reduces to:
  $$\text{Optimized Time} = \max(T_{\text{DataFetcher}}, T_{\text{NewsAnalysis}}, T_{\text{RiskAssessment}}) \approx 5.0\text{s}$$
  *Net latency reduction: ~48%.*

---

## 4. Reliability & Cache Architecture

1. **Caching**: 1-hour TTL (`TOOL_CACHE_TTL_SECONDS=3600`) using Redis if connected, otherwise transparent in-memory fallback.
2. **Exponential Backoff**: Up to 3 attempts ($2^0=1\text{s}, 2^1=2\text{s}$) for transient tool failures before halting.
3. **Audit Logging**: Structured single-line JSON logs emitted for every tool invocation capturing `tool_name`, `input`, `output`, `cached`, `latency_ms`, and `timestamp`.
4. **Human Verification Barrier**: Explicit REST confirmation required before `ReportWriterAgent` generates the final markdown artifact.

---

## 5. Container Specification

- **Base Image**: `python:3.11.9-slim-bookworm` (Multi-stage build with non-root `appuser` UID 10001).
- **Service Name**: `agent_orchestrator` (mapped to port `8003:8000`).
- **Health Check**: Automated periodic HTTP probe against `/health`.
