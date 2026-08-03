"""Metrics shared by transformer classification and summarization."""

from __future__ import annotations

import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, classification_report, f1_score


def classification_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
    }


def detailed_classification_metrics(labels: list[int], predictions: list[int]) -> dict:
    return classification_report(
        labels,
        predictions,
        target_names=["negative", "neutral", "positive"],
        output_dict=True,
        zero_division=0,
    )


def rouge_metrics(references: list[str], predictions: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    values = [
        scorer.score(reference, prediction)
        for reference, prediction in zip(references, predictions, strict=True)
    ]
    return {
        name: float(np.mean([score[name].fmeasure for score in values]))
        for name in ("rouge1", "rouge2", "rougeL")
    }
