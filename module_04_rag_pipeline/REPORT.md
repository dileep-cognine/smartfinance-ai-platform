# Module 4 Report - Retrieval-Augmented Generation

## Current Implementation

The API is backend-only, validates session IDs and prompt length, retains the
last five turns per session, preserves source/chunk metadata, and returns
`Insufficient information` for weak retrieval. Three chunking strategies are
implemented. A deterministic TF-IDF retrieval baseline keeps health checks and
local development functional without provider keys.

## Required Experiments Still To Run

- Replace semantic chunking vectors with `all-MiniLM-L6-v2`, then record chunk
  count and average size for all three strategies on the same filing.
- Build and persist FAISS flat-L2 and ChromaDB stores over identical chunks.
- Evaluate 20 queries, recording latency and manually judged top-5 hit rate.
- Add LangChain MMR and compare redundancy.
- Configure OpenAI or local Ollama generation with the same grounding contract.
- Run the 15-pair evaluation and three multi-turn transcripts. Report measured
  faithfulness, context relevance, and answer relevance.

Base image: `python:3.11.9-slim-bookworm`, appropriate for a CPU FastAPI
service. Record final image size after building.
