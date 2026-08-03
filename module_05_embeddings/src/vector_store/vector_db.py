"""Thread-safe local semantic-search baseline."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class SimilarDocument:
    document_id: str
    text: str
    score: float


class DocumentIndex:
    def __init__(self, documents: list[tuple[str, str]]) -> None:
        if not documents:
            raise ValueError("documents cannot be empty")
        self._documents = documents
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(text for _, text in documents)

    @property
    def size(self) -> int:
        return len(self._documents)

    def search(self, text: str, top_k: int) -> list[SimilarDocument]:
        if not text.strip():
            raise ValueError("document text cannot be blank")
        scores = cosine_similarity(self._vectorizer.transform([text]), self._matrix)[0]
        return [
            SimilarDocument(
                document_id=self._documents[int(index)][0],
                text=self._documents[int(index)][1],
                score=float(scores[int(index)]),
            )
            for index in scores.argsort()[::-1][:top_k]
        ]
