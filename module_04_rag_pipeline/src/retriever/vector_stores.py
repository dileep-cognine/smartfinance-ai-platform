"""FAISS, Chroma, and MMR retrieval over identical dense embeddings."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..chunking import Chunk


@dataclass(frozen=True)
class DenseResult:
    chunk: Chunk
    score: float


class DenseEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")


class FaissVectorStore:
    def __init__(self, chunks: list[Chunk], encoder: DenseEncoder) -> None:
        self.chunks = chunks
        self.encoder = encoder
        self.vectors = encoder.encode([chunk.text for chunk in chunks])
        self.index = faiss.IndexFlatL2(self.vectors.shape[1])
        self.index.add(self.vectors)

    def search(self, query: str, top_k: int = 5) -> list[DenseResult]:
        distances, indices = self.index.search(self.encoder.encode([query]), top_k)
        return [
            DenseResult(self.chunks[int(index)], float(1.0 / (1.0 + distance)))
            for distance, index in zip(distances[0], indices[0], strict=True)
            if index >= 0
        ]

    def mmr_search(
        self, query: str, top_k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.7
    ) -> list[DenseResult]:
        query_vector = self.encoder.encode([query])[0]
        _, candidate_indices = self.index.search(query_vector.reshape(1, -1), fetch_k)
        candidates = [int(index) for index in candidate_indices[0] if index >= 0]
        selected: list[int] = []
        while candidates and len(selected) < top_k:
            best_index = max(
                candidates,
                key=lambda index: lambda_mult * float(self.vectors[index] @ query_vector)
                - (1 - lambda_mult)
                * max(
                    [float(self.vectors[index] @ self.vectors[chosen]) for chosen in selected]
                    or [0.0]
                ),
            )
            selected.append(best_index)
            candidates.remove(best_index)
        return [
            DenseResult(self.chunks[index], float(self.vectors[index] @ query_vector))
            for index in selected
        ]


class ChromaVectorStore:
    def __init__(
        self,
        chunks: list[Chunk],
        encoder: DenseEncoder,
        persist_directory: Path,
        collection_name: str = "sec_filings",
    ) -> None:
        self.chunks = chunks
        self.encoder = encoder
        client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if self.collection.count() != len(chunks):
            if self.collection.count():
                client.delete_collection(collection_name)
                self.collection = client.create_collection(
                    collection_name, metadata={"hnsw:space": "cosine"}
                )
            vectors = encoder.encode([chunk.text for chunk in chunks])
            self.collection.add(
                ids=[f"{chunk.source}:{chunk.number}" for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=vectors.tolist(),
                metadatas=[
                    {"source": chunk.source, "number": chunk.number} for chunk in chunks
                ],
            )

    def search(self, query: str, top_k: int = 5) -> list[DenseResult]:
        result = self.collection.query(
            query_embeddings=self.encoder.encode([query]).tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            DenseResult(
                Chunk(document, metadata["source"], int(metadata["number"])),
                float(1.0 - distance),
            )
            for document, metadata, distance in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]


def benchmark_store(store, queries: list[str], top_k: int = 5) -> dict:
    latencies = []
    results = []
    for query in queries:
        started = time.perf_counter()
        found = store.search(query, top_k)
        latencies.append((time.perf_counter() - started) * 1000)
        results.append(
            {
                "query": query,
                "results": [
                    {"score": item.score, **asdict(item.chunk)} for item in found
                ],
            }
        )
    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "queries": results,
    }
