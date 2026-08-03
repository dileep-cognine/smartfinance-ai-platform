from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    TARGET,
    fit_preprocessor,
    load_credit_data,
    split_raw_data,
    transform,
)
from src.metrics import classification_metrics, select_threshold


@pytest.fixture
def credit_frame() -> pd.DataFrame:
    rows = []
    for index in range(100):
        rows.append(
            {
                TARGET: int(index % 5 == 0),
                "age": 20 + index,
                "monthly_income": None if index % 9 == 0 else 2000 + index,
                "debt_ratio": index / 100,
                "late_30_59_days": index % 3,
                "late_60_89_days": index % 2,
                "late_90_days": index % 4,
            }
        )
    return pd.DataFrame(rows)


def test_split_is_reproducible_and_stratified(credit_frame: pd.DataFrame) -> None:
    first = split_raw_data(credit_frame)
    second = split_raw_data(credit_frame)
    assert {name: len(frame) for name, frame in first.items()} == {
        "train": 70,
        "validation": 15,
        "test": 15,
    }
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
        assert first[name][TARGET].mean() == pytest.approx(0.2, abs=0.04)


def test_transform_removes_missing_and_engineers_features(credit_frame: pd.DataFrame) -> None:
    splits = split_raw_data(credit_frame)
    result = transform(splits["validation"], fit_preprocessor(splits["train"]))
    assert not result.isna().any().any()
    assert {"total_late_payments", "debt_income_ratio"}.issubset(result.columns)


def test_threshold_prefers_recall_under_precision_constraint() -> None:
    truth = [1, 1, 0, 0]
    probabilities = [0.9, 0.8, 0.7, 0.1]
    threshold = select_threshold(truth, probabilities, minimum_precision=0.70)
    metrics = classification_metrics(truth, probabilities, threshold)
    assert metrics["precision"] >= 0.70
    assert metrics["recall"] == 1.0


def test_loader_rejects_missing_target(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"age": [30]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="target"):
        load_credit_data(path)
