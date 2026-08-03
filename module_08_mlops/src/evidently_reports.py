"""Generate Evidently reports for five progressively shifted periods."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def simulate_period(reference: pd.DataFrame, period: int, seed: int = 42) -> pd.DataFrame:
    generator = np.random.default_rng(seed + period)
    current = reference.sample(
        frac=1, replace=True, random_state=seed + period
    ).reset_index(drop=True)
    numeric = current.select_dtypes(include=np.number).columns
    magnitude = period * 0.08
    for column in numeric:
        spread = float(reference[column].std() or 1.0)
        current[column] = current[column] + magnitude * spread + generator.normal(
            0, magnitude * spread, len(current)
        )
    return current


def generate_five_periods(reference: pd.DataFrame, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for period in range(1, 6):
        current = simulate_period(reference, period)
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(output_dir / f"period_{period}.html"))
        result = report.as_dict()
        (output_dir / f"period_{period}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        summaries.append({"period": period, "report": result})
    return summaries
