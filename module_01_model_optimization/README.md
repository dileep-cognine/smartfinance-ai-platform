# Module 1 - Credit-Risk Model Optimization

This module implements leakage-safe preprocessing and the three required
XGBoost hyperparameter-search strategies. The held-out test partition is used
only after model and decision-threshold selection.

```bash
python -m src.main prepare-data
python -m src.main tune --strategy baseline --no-mlflow
python -m src.main tune --strategy grid --no-mlflow
python -m src.main tune --strategy random --trials 50
python -m src.main tune --strategy bayesian --trials 100
```

Omit `--no-mlflow` when the tracking server is available. Outputs include raw
CV results, plots, JSON summaries, and serialized models.
