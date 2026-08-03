"""Prediction adapter that always resolves the current champion alias."""

from __future__ import annotations

import pandas as pd

from ..model_registry.registry import load_production_model


def predict(frame: pd.DataFrame):
    model = load_production_model()
    return model.predict(frame)
