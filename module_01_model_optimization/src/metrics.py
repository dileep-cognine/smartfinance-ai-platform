"""Evaluation and threshold-selection utilities."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    minimum_precision: float = 0.70,
) -> float:
    """Maximize recall among thresholds satisfying the precision constraint."""
    truth = np.asarray(list(y_true))
    scores = np.asarray(list(probabilities))
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    candidates = [
        (float(recall[index]), float(threshold))
        for index, threshold in enumerate(thresholds)
        if precision[index] >= minimum_precision
    ]
    if candidates:
        return max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))[1]

    f2_values = [
        fbeta_score(truth, scores >= threshold, beta=2, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(f2_values))]) if len(thresholds) else 0.5


def classification_metrics(
    y_true: Iterable[int], probabilities: Iterable[float], threshold: float
) -> dict[str, float]:
    """Calculate the business and model-selection metrics."""
    truth = np.asarray(list(y_true))
    scores = np.asarray(list(probabilities))
    predictions = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f2": float(fbeta_score(truth, predictions, beta=2, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "threshold": float(threshold),
    }
