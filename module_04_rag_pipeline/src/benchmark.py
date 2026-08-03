"""Build both vector stores and persist a 20-query comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import fixed_chunks, semantic_chunks, sentence_chunks
from .retriever.vector_stores import (
    ChromaVectorStore,
    DenseEncoder,
    FaissVectorStore,
    benchmark_store,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filings", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chunks = []
    documents = sorted(args.filings.glob("*_cleaned.txt"))
    for path in documents:
        chunks.extend(fixed_chunks(path.read_text(encoding="utf-8"), path.name))
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    if len(queries) != 20:
        raise ValueError("The retrieval benchmark requires exactly 20 queries")
    encoder = DenseEncoder()
    faiss_store = FaissVectorStore(chunks, encoder)
    chroma_store = ChromaVectorStore(chunks, encoder, args.output / "chroma")
    result = {
        "chunking": {},
        "faiss": benchmark_store(faiss_store, queries),
        "chroma": benchmark_store(chroma_store, queries),
        "faiss_mmr": [
            {
                "query": query,
                "results": [
                    {
                        "source": item.chunk.source,
                        "chunk_number": item.chunk.number,
                        "score": item.score,
                    }
                    for item in faiss_store.mmr_search(query)
                ],
            }
            for query in queries
        ],
        "manual_hit_rate_instructions": (
            "Add relevant=true/false to each returned chunk, then calculate top-5 hit rate."
        ),
    }
    if documents:
        text = documents[0].read_text(encoding="utf-8")
        for name, produced in {
            "fixed_512_overlap_50": fixed_chunks(text, documents[0].name),
            "sentence": sentence_chunks(text, documents[0].name),
            "semantic": semantic_chunks(text, documents[0].name),
        }.items():
            sizes = [len(chunk.text.split()) for chunk in produced]
            result["chunking"][name] = {
                "count": len(produced),
                "average_words": sum(sizes) / len(sizes) if sizes else 0,
            }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "retrieval_benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
