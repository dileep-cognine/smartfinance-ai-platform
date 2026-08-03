"""Local retrieval baseline with source-preserving results."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..chunking import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class LocalVectorStore:
    """In-memory index used when external vector stores are unavailable."""

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=50_000)
        self._matrix = self._vectorizer.fit_transform(chunk.text for chunk in chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query cannot be blank")
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix)[0]
        indices = scores.argsort()[::-1][:top_k]
        return [
            SearchResult(self._chunks[int(index)], float(scores[int(index)]))
            for index in indices
            if scores[int(index)] > 0
        ]
