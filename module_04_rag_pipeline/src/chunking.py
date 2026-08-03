"""Document chunking strategies required by the RAG assessment."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    number: int


def fixed_chunks(text: str, source: str, size: int = 512, overlap: int = 50) -> list[Chunk]:
    """Split on words using the required 512/50 defaults."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")
    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        content = " ".join(words[start : start + size]).strip()
        if content:
            chunks.append(Chunk(content, source, len(chunks)))
        if start + size >= len(words):
            break
    return chunks


def sentence_chunks(text: str, source: str, max_words: int = 512) -> list[Chunk]:
    """Group sentence boundaries without downloading tokenizer state."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > max_words:
            chunks.append(Chunk(" ".join(current), source, len(chunks)))
            current, current_words = [], 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append(Chunk(" ".join(current), source, len(chunks)))
    return chunks


def semantic_chunks(text: str, source: str, threshold: float = 0.25) -> list[Chunk]:
    """Split where adjacent sentence TF-IDF similarity falls below a threshold.

    This deterministic baseline keeps the API usable offline. The assessment
    experiment must replace the TF-IDF vectors with the configured sentence
    transformer and report both configurations.
    """
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], source, 0)]
    vectors = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    chunks: list[Chunk] = []
    current = [sentences[0]]
    for index in range(1, len(sentences)):
        similarity = float(vectors[index - 1].multiply(vectors[index]).sum())
        if not np.isfinite(similarity) or similarity < threshold:
            chunks.append(Chunk(" ".join(current), source, len(chunks)))
            current = []
        current.append(sentences[index])
    if current:
        chunks.append(Chunk(" ".join(current), source, len(chunks)))
    return chunks
