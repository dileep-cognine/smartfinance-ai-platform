"""Fifteen-pair RAG evaluation with transparent lexical proxy metrics."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import httpx


def _terms(text: str) -> Counter:
    return Counter(re.findall(r"[a-z]{3,}", text.lower()))


def overlap(left: str, right: str) -> float:
    left_terms, right_terms = _terms(left), _terms(right)
    if not left_terms:
        return 0.0
    return sum((left_terms & right_terms).values()) / sum(left_terms.values())


def evaluate_answer(question: str, answer: str, contexts: list[str]) -> dict[str, float]:
    joined = " ".join(contexts)
    return {
        "faithfulness": overlap(answer, joined),
        "context_relevance": overlap(question, joined),
        "answer_relevance": overlap(question, answer),
    }


def evaluate_suite(endpoint: str, qa_path: Path, output_path: Path) -> dict:
    pairs = json.loads(qa_path.read_text(encoding="utf-8"))
    if len(pairs) != 15:
        raise ValueError("RAG evaluation requires exactly 15 question-answer pairs")
    rows = []
    with httpx.Client(timeout=60) as client:
        for index, pair in enumerate(pairs):
            response = client.post(
                endpoint,
                json={
                    "question": pair["question"],
                    "session_id": f"evaluation-{index}",
                    "top_k": 5,
                },
            )
            response.raise_for_status()
            payload = response.json()
            contexts = [
                citation["text"]
                for citation in payload["citations"]
            ]
            metrics = evaluate_answer(pair["question"], payload["answer"], contexts)
            metrics["reference_overlap"] = overlap(pair["answer"], payload["answer"])
            rows.append({"question": pair["question"], "response": payload, "metrics": metrics})
    aggregate = {
        metric: sum(row["metrics"][metric] for row in rows) / len(rows)
        for metric in rows[0]["metrics"]
    }
    result = {"aggregate": aggregate, "cases": rows}
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
