"""Age-group fairness audit and reduction-based mitigation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from fairlearn.reductions import DemographicParity, ExponentiatedGradient, GridSearch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler


def age_groups(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 30, 45, 60, 120], labels=["18-30", "31-45", "46-60", "61+"])


def fairness_metrics(
    truth: pd.Series, prediction, sensitive: pd.Series
) -> dict[str, float | dict]:
    frame = MetricFrame(
        metrics={"accuracy": accuracy_score, "recall": recall_score, "selection": selection_rate},
        y_true=truth,
        y_pred=prediction,
        sensitive_features=sensitive,
    )
    rates = frame.by_group["selection"]
    return {
        "demographic_parity_difference": float(
            demographic_parity_difference(truth, prediction, sensitive_features=sensitive)
        ),
        "equalized_odds_difference": float(
            equalized_odds_difference(truth, prediction, sensitive_features=sensitive)
        ),
        "disparate_impact_ratio": float(rates.min() / rates.max()) if rates.max() else 0.0,
        "by_group": frame.by_group.to_dict(),
    }


def train_fair_models(train_path: Path, test_path: Path, output_path: Path) -> dict:
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    target = "target_default"
    features = [column for column in train if column != target]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features])
    x_test = scaler.transform(test[features])
    sensitive_train, sensitive_test = age_groups(train["age"]), age_groups(test["age"])
    constraint = DemographicParity()
    estimators = {
        "unconstrained": LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            x_train, train[target]
        ),
        "exponentiated_gradient": ExponentiatedGradient(
            LogisticRegression(max_iter=1000), constraint
        ).fit(x_train, train[target], sensitive_features=sensitive_train),
        "grid_search": GridSearch(
            LogisticRegression(max_iter=1000), constraint, grid_size=20
        ).fit(x_train, train[target], sensitive_features=sensitive_train),
    }
    results = {}
    points = []
    for name, estimator in estimators.items():
        prediction = estimator.predict(x_test)
        results[name] = {
            "accuracy": float(accuracy_score(test[target], prediction)),
            **fairness_metrics(test[target], prediction, sensitive_test),
        }
        points.append(
            (
                results[name]["demographic_parity_difference"],
                results[name]["accuracy"],
                name,
            )
        )
    for difference, accuracy, name in points:
        plt.scatter(difference, accuracy, label=name)
    plt.xlabel("Demographic parity difference")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return results
