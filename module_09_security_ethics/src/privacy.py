"""PII audit and differentially private credit-model comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from diffprivlib.models import LogisticRegression as DPLogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .guardrails import PII_PATTERNS, anonymize_pii


def audit_texts(texts: list[str]) -> list[dict]:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
    findings = []
    for index, text in enumerate(texts):
        for label, pattern in PII_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(
                    {"row": index, "type": label, "start": match.start(), "end": match.end()}
                )
        for entity in nlp(text).ents:
            if entity.label_ in {"PERSON", "GPE", "ORG"}:
                findings.append(
                    {
                        "row": index,
                        "type": entity.label_,
                        "start": entity.start_char,
                        "end": entity.end_char,
                    }
                )
    return findings


def anonymize_frame(frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    result = frame.copy()
    result[text_column] = result[text_column].astype(str).map(anonymize_pii)
    return result


def compare_differential_privacy(
    train_path: Path,
    test_path: Path,
    epsilon: float = 1.0,
) -> dict[str, dict[str, float]]:
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    target = "target_default"
    features = [column for column in train.columns if column != target]
    x_train, y_train = train[features], train[target]
    x_test, y_test = test[features], test[target]
    preprocessing = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    x_train_scaled = preprocessing.fit_transform(x_train)
    x_test_scaled = preprocessing.transform(x_test)
    bound = float(np.linalg.norm(x_train_scaled, axis=1).max())
    models = {
        "standard": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "differential_private": DPLogisticRegression(
            epsilon=epsilon,
            data_norm=bound,
            max_iter=1000,
        ),
    }
    results = {}
    for name, model in models.items():
        prediction = model.fit(x_train_scaled, y_train).predict(x_test_scaled)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "f1": float(f1_score(y_test, prediction)),
            "epsilon": epsilon if name == "differential_private" else float("inf"),
        }
    return results
