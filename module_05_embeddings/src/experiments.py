"""Embedding comparison, search evaluation, duplicates, and clustering using Hugging Face models."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity

LOCAL_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
}


@dataclass
class EmbeddingRun:
    name: str
    vectors: np.ndarray
    latency_ms_per_document: float


def load_corpus(path: Path, limit: int = 200) -> pd.DataFrame:
    frame = pd.read_csv(path).dropna(subset=["Sentence", "Sentiment"])
    parts = []
    per_class = max(1, limit // frame["Sentiment"].nunique())
    for _, group in frame.groupby("Sentiment"):
        parts.append(group.sample(min(per_class, len(group)), random_state=42))
    return pd.concat(parts).sample(frac=1, random_state=42).head(limit).reset_index(drop=True)


def embed_local(name: str, texts: list[str]) -> EmbeddingRun:
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    model = SentenceTransformer(LOCAL_MODELS[name], token=token) if token else SentenceTransformer(LOCAL_MODELS[name])
    started = time.perf_counter()
    vectors = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
    )
    return EmbeddingRun(name, np.asarray(vectors), (time.perf_counter() - started) * 1000 / len(texts))


def save_umap(run: EmbeddingRun, labels: list[str], path: Path) -> None:
    coordinates = umap.UMAP(n_components=2, random_state=42).fit_transform(run.vectors)
    frame = pd.DataFrame({"x": coordinates[:, 0], "y": coordinates[:, 1], "label": labels})
    for label, group in frame.groupby("label"):
        plt.scatter(group["x"].values, group["y"].values, label=str(label), alpha=0.75)
    plt.legend()
    plt.title(f"UMAP: {run.name}")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def precision_at_five(
    query_vectors: np.ndarray,
    document_vectors: np.ndarray,
    query_labels: list[str],
    document_labels: list[str],
) -> float:
    similarities = cosine_similarity(query_vectors, document_vectors)
    values = []
    for index, scores in enumerate(similarities):
        top = scores.argsort()[::-1][:5]
        values.append(sum(document_labels[item] == query_labels[index] for item in top) / 5)
    return float(np.mean(values))


def duplicate_pairs(vectors: np.ndarray, threshold: float = 0.92) -> list[tuple[int, int, float]]:
    similarities = cosine_similarity(vectors)
    rows, columns = np.where(np.triu(similarities, k=1) >= threshold)
    return [
        (int(row), int(column), float(similarities[row, column]))
        for row, column in zip(rows, columns, strict=True)
    ]


def cluster_metrics(vectors: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    kmeans_labels = KMeans(n_clusters=8, random_state=42, n_init=20).fit_predict(vectors)
    hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=10, metric="euclidean").fit_predict(vectors)
    results: dict[str, Any] = {
        "kmeans": {
            "silhouette": float(silhouette_score(vectors, kmeans_labels)),
            "davies_bouldin": float(davies_bouldin_score(vectors, kmeans_labels)),
            "clusters": int(len(set(kmeans_labels))),
        }
    }
    valid = hdbscan_labels >= 0
    if valid.sum() > 2 and len(set(hdbscan_labels[valid])) > 1:
        results["hdbscan"] = {
            "silhouette": float(silhouette_score(vectors[valid], hdbscan_labels[valid])),
            "davies_bouldin": float(
                davies_bouldin_score(vectors[valid], hdbscan_labels[valid])
            ),
            "clusters": int(len(set(hdbscan_labels[valid]))),
            "noise_documents": int((~valid).sum()),
        }
    else:
        results["hdbscan"] = {"status": "insufficient_non_noise_clusters"}
    return results, kmeans_labels


def cluster_keywords(texts: list[str], labels: np.ndarray, terms: int = 6) -> dict[str, list[str]]:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(texts)
    names = np.asarray(vectorizer.get_feature_names_out())
    output: dict[str, list[str]] = {}
    for label in sorted(set(labels)):
        indices = np.where(labels == label)[0]
        scores = np.asarray(matrix[indices].mean(axis=0)).ravel()
        output[str(int(label))] = names[scores.argsort()[::-1][:terms]].tolist()
    return output


def run_experiments(corpus_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(corpus_path)
    texts = corpus["Sentence"].tolist()
    labels = corpus["Sentiment"].tolist()
    runs = [
        embed_local("minilm", texts),
        embed_local("mpnet", texts),
        embed_local("bge_small", texts),
    ]
    summary = {}
    for run in runs:
        np.save(output_dir / f"{run.name}.npy", run.vectors)
        save_umap(run, labels, output_dir / f"{run.name}_umap.png")
        queries = run.vectors[:10]
        clustering, cluster_labels = cluster_metrics(run.vectors)
        summary[run.name] = {
            "latency_ms_per_document": run.latency_ms_per_document,
            "dimension": int(run.vectors.shape[1]),
            "estimated_50k_storage_mb_float32": float(50_000 * run.vectors.shape[1] * 4 / 2**20),
            "precision_at_5_label_proxy": precision_at_five(
                queries, run.vectors[10:], labels[:10], labels[10:]
            ),
            "duplicate_pairs_at_0_92": duplicate_pairs(run.vectors),
            "clustering": clustering,
            "kmeans_keywords": cluster_keywords(texts, cluster_labels),
        }
    (output_dir / "embedding_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run_experiments(arguments.corpus, arguments.output)
