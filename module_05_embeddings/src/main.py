"""Semantic-search API for the assessment document corpus."""

from __future__ import annotations

import csv
import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .vector_store.vector_db import DocumentIndex

app = FastAPI(title="SmartFinance Embeddings API", version="0.1.0")
_index: DocumentIndex | None = None


class SimilarDocsRequest(BaseModel):
    document_text: str = Field(min_length=2, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=25)


class SimilarDocsResponse(BaseModel):
    model: str
    results: list[dict[str, str | float]]


def _load_documents() -> list[tuple[str, str]]:
    path = (
        Path(os.getenv("DATA_DIR", "data"))
        / "processed/cleaned/financial_phrasebank_clean.csv"
    )
    if not path.exists():
        return [("system-0", "No document corpus has been mounted.")]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            (f"phrasebank-{index}", row["Sentence"])
            for index, row in enumerate(reader)
            if row.get("Sentence")
        ]


def get_index() -> DocumentIndex:
    global _index
    if _index is None:
        _index = DocumentIndex(_load_documents())
    return _index


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "healthy", "documents": get_index().size}


@app.post("/similar-docs", response_model=SimilarDocsResponse)
def similar_docs(request: SimilarDocsRequest) -> SimilarDocsResponse:
    results = get_index().search(request.document_text, request.top_k)
    return SimilarDocsResponse(
        model="tfidf-baseline",
        results=[
            {
                "document_id": result.document_id,
                "text": result.text,
                "score": result.score,
            }
            for result in results
        ],
    )
