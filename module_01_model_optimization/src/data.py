"""Leakage-safe preparation of the Give Me Some Credit dataset."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "target_default"
RANDOM_SEED = 42


@dataclass(frozen=True)
class PreprocessingState:
    """Statistics fitted only on the training partition."""

    medians: dict[str, float]
    lower_bounds: dict[str, float]
    upper_bounds: dict[str, float]


def load_credit_data(path: Path) -> pd.DataFrame:
    """Load and validate the assessment credit-risk data."""
    frame = pd.read_csv(path, na_values=["NA", ""])
    frame = frame.rename(columns=lambda name: name.strip().lower())
    frame = frame.drop(columns=["", "unnamed: 0"], errors="ignore")
    if TARGET not in frame:
        raise ValueError(f"Required target column '{TARGET}' is missing")
    if not set(frame[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("target_default must contain only binary labels")
    if frame[TARGET].isna().any():
        raise ValueError("target_default contains missing values")
    if frame.drop(columns=TARGET).empty:
        raise ValueError("Credit dataset has no feature columns")
    return frame


def split_raw_data(
    frame: pd.DataFrame, random_seed: int = RANDOM_SEED
) -> dict[str, pd.DataFrame]:
    """Create deterministic 70/15/15 stratified partitions."""
    train, remainder = train_test_split(
        frame,
        test_size=0.30,
        random_state=random_seed,
        stratify=frame[TARGET],
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_seed,
        stratify=remainder[TARGET],
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def fit_preprocessor(train: pd.DataFrame) -> PreprocessingState:
    """Fit median imputation and 1st/99th percentile clipping statistics."""
    features = train.drop(columns=TARGET).apply(pd.to_numeric, errors="coerce")
    medians = features.median().to_dict()
    imputed = features.fillna(medians)
    return PreprocessingState(
        medians={key: float(value) for key, value in medians.items()},
        lower_bounds={key: float(value) for key, value in imputed.quantile(0.01).items()},
        upper_bounds={key: float(value) for key, value in imputed.quantile(0.99).items()},
    )


def transform(frame: pd.DataFrame, state: PreprocessingState) -> pd.DataFrame:
    """Apply fitted preprocessing statistics and deterministic feature engineering."""
    labels = frame[TARGET].astype(int).reset_index(drop=True)
    features = frame.drop(columns=TARGET).apply(pd.to_numeric, errors="coerce")
    features = features.fillna(state.medians)
    for column in features.columns:
        features[column] = features[column].clip(
            lower=state.lower_bounds[column],
            upper=state.upper_bounds[column],
        )

    late_columns = ["late_30_59_days", "late_60_89_days", "late_90_days"]
    if set(late_columns).issubset(features):
        features["total_late_payments"] = features[late_columns].sum(axis=1)
    if {"debt_ratio", "monthly_income"}.issubset(features):
        features["debt_income_ratio"] = (
            features["debt_ratio"] / (features["monthly_income"] + 1.0)
        )
    features[TARGET] = labels
    return features


def prepare_and_save(
    input_path: Path, output_dir: Path, random_seed: int = RANDOM_SEED
) -> dict[str, Path]:
    """Create processed partitions and a machine-readable preprocessing record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_splits = split_raw_data(load_credit_data(input_path), random_seed)
    state = fit_preprocessor(raw_splits["train"])

    paths: dict[str, Path] = {}
    for name, raw_frame in raw_splits.items():
        processed = transform(raw_frame, state)
        if processed.isna().any().any():
            raise ValueError(f"Preprocessing left missing values in the {name} split")
        path = output_dir / f"credit_{name}.csv"
        processed.to_csv(path, index=False)
        paths[name] = path

    metadata = {
        "source": str(input_path),
        "random_seed": random_seed,
        "target": TARGET,
        "row_counts": {name: len(frame) for name, frame in raw_splits.items()},
        "positive_rates": {
            name: float(frame[TARGET].mean()) for name, frame in raw_splits.items()
        },
        "preprocessing": asdict(state),
    }
    metadata_path = output_dir / "preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["metadata"] = metadata_path
    return paths


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a processed split as features and labels."""
    frame = pd.read_csv(path)
    if frame.isna().any().any():
        raise ValueError(f"Processed split contains missing values: {path}")
    return frame.drop(columns=TARGET), frame[TARGET].astype(int)
