"""Population Stability Index calculations and alert decisions."""

from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: list[float], current: list[float], bins: int = 10
) -> float:
    """Calculate PSI with reference-derived quantile buckets."""
    expected = np.asarray(reference, dtype=float)
    actual = np.asarray(current, dtype=float)
    if expected.size < bins or actual.size < bins:
        raise ValueError("Each sample must contain at least as many values as bins")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Samples must contain only finite numeric values")
    boundaries = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(boundaries) < 3:
        return 0.0 if np.allclose(expected.mean(), actual.mean()) else 1.0
    boundaries[0], boundaries[-1] = -np.inf, np.inf
    expected_counts, _ = np.histogram(expected, bins=boundaries)
    actual_counts, _ = np.histogram(actual, bins=boundaries)
    expected_share = np.clip(expected_counts / expected.size, 1e-6, None)
    actual_share = np.clip(actual_counts / actual.size, 1e-6, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))
