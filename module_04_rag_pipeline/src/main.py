"""Grounded, session-aware filing Q&A API."""

from __future__ import annotations

import os
import re
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .chunking import Chunk, fixed_chunks
from .generator.llm import generate_with_langchain
from .retriever.store import LocalVectorStore

app = FastAPI(title="SmartFinance RAG API", version="0.1.0")
_memory: dict[str, deque[tuple[str, str]]] = defaultdict(
    lambda: deque(maxlen=int(os.getenv("RAG_MEMORY_TURNS", "5")))
)
_memory_lock = Lock()
_store: LocalVectorStore | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    top_k: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    source: str
    chunk_number: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    session_id: str


def _load_store() -> LocalVectorStore:
    global _store
    if _store is not None:
        return _store
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    filing_dir = data_dir / "processed/cleaned/sec_filings"
    chunks: list[Chunk] = []
    for path in sorted(filing_dir.glob("*_cleaned.txt")):
        chunks.extend(fixed_chunks(path.read_text(encoding="utf-8", errors="ignore"), path.name))
    if not chunks:
        chunks = [Chunk("No filing corpus has been mounted.", "system", 0)]
    _store = LocalVectorStore(chunks)
    return _store


def _grounded_answer(question: str, results: list) -> str:
    if not results or results[0].score < 0.02:
        return "Insufficient information"
    query_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z]{3,}", question)
        if term.lower() not in {"what", "when", "where", "which", "about", "from"}
    }
    sentences: list[str] = []
    for result in results:
        for sentence in re.split(r"(?<=[.!?])\s+", result.chunk.text):
            if query_terms.intersection(re.findall(r"[a-z]{3,}", sentence.lower())):
                citation = f"[{result.chunk.source}#{result.chunk.number}]"
                sentences.append(f"{sentence.strip()} {citation}")
                break
        if len(sentences) == 3:
            break
    return " ".join(sentences) if sentences else "Insufficient information"


@app.get("/health")
def health() -> dict[str, str]:
    _load_store()
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    results = _load_store().search(request.question, request.top_k)
    with _memory_lock:
        history = "\n".join(
            f"Q: {question}\nA: {answer}"
            for question, answer in _memory.get(request.session_id, [])
        )
    context = "\n\n".join(
        f"[{result.chunk.source}#{result.chunk.number}] {result.chunk.text}"
        for result in results
    )
    answer = generate_with_langchain(request.question, context, history)
    if not answer:
        answer = _grounded_answer(request.question, results)
    with _memory_lock:
        _memory[request.session_id].append((request.question, answer))
    return QueryResponse(
        answer=answer,
        citations=[
            Citation(
                source=result.chunk.source,
                chunk_number=result.chunk.number,
                score=result.score,
                text=result.chunk.text,
            )
            for result in results
        ],
        session_id=request.session_id,
    )


@app.get("/sessions/{session_id}")
def session_history(session_id: str) -> dict[str, list[dict[str, str]]]:
    with _memory_lock:
        history = list(_memory.get(session_id, []))
    return {"turns": [{"question": question, "answer": answer} for question, answer in history]}
