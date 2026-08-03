"""Grid, random, and Bayesian XGBoost optimization."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import ConfusionMatrixDisplay, fbeta_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from xgboost import XGBClassifier

from .metrics import classification_metrics, select_threshold
from .tracking import log_run, tracked_run

F2_SCORER = make_scorer(fbeta_score, beta=2, zero_division=0)


@dataclass
class TuningResult:
    strategy: str
    best_score: float
    elapsed_seconds: float
    models_trained: int
    best_params: dict[str, Any]
    threshold: float
    test_metrics: dict[str, float]
    model_path: Path
    mlflow_run_id: str | None


def build_estimator(random_seed: int = 42, **parameters: Any) -> XGBClassifier:
    """Create a deterministic binary XGBoost classifier."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_seed,
        n_jobs=-1,
        tree_method="hist",
        **parameters,
    )


def _save_grid_heatmap(results: pd.DataFrame, path: Path) -> None:
    grouped = (
        results.groupby(["param_max_depth", "param_learning_rate"], as_index=False)[
            "mean_test_score"
        ]
        .max()
        .pivot(
            index="param_max_depth",
            columns="param_learning_rate",
            values="mean_test_score",
        )
    )
    plt.figure(figsize=(7, 5))
    sns.heatmap(grouped, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Grid search: best F2 by depth and learning rate")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_parallel_coordinates(results: pd.DataFrame, path: Path) -> None:
    columns = [
        "param_n_estimators",
        "param_max_depth",
        "param_learning_rate",
        "param_subsample",
        "param_colsample_bytree",
        "mean_test_score",
    ]
    plot_data = results[columns].astype(float).copy()
    normalized = (plot_data - plot_data.min()) / (
        plot_data.max() - plot_data.min()
    ).replace(0, 1)
    score_min = plot_data["mean_test_score"].min()
    score_range = plot_data["mean_test_score"].max() - score_min or 1.0
    plt.figure(figsize=(11, 6))
    for index, row in normalized.iterrows():
        color = plt.cm.viridis((plot_data.loc[index, "mean_test_score"] - score_min) / score_range)
        plt.plot(range(len(columns)), row, color=color, alpha=0.35)
    labels = [name.replace("param_", "") for name in columns]
    plt.xticks(range(len(columns)), labels, rotation=20)
    plt.ylabel("Normalized value")
    plt.title("Random search parameter relationships")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_optuna_plots(study: optuna.Study, output_dir: Path) -> list[Path]:
    history_path = output_dir / "bayesian_optimization_history.png"
    importance_path = output_dir / "bayesian_parameter_importance.png"

    completed = [trial for trial in study.trials if trial.value is not None]
    values = [trial.value for trial in completed]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(values) + 1), values, marker=".", linewidth=1)
    plt.xlabel("Completed trial")
    plt.ylabel("1 - mean CV F2")
    plt.title("Bayesian optimization history")
    plt.tight_layout()
    plt.savefig(history_path, dpi=160)
    plt.close()

    try:
        importance = optuna.importance.get_param_importances(study)
    except (RuntimeError, ValueError):
        importance = {}
    plt.figure(figsize=(8, 5))
    plt.barh(list(importance), list(importance.values()))
    plt.xlabel("Importance")
    plt.title("Bayesian hyperparameter importance")
    plt.tight_layout()
    plt.savefig(importance_path, dpi=160)
    plt.close()
    return [history_path, importance_path]


def tune(
    strategy: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    trials: int | None = None,
    random_seed: int = 42,
    mlflow_enabled: bool = True,
) -> TuningResult:
    """Run one required search strategy and evaluate once on held-out test data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    estimator = build_estimator(random_seed)
    artifact_paths: list[Path] = []
    started = time.perf_counter()

    if strategy == "baseline":
        model = estimator
        model.fit(x_train, y_train)
        best_score = float(
            fbeta_score(
                y_validation,
                model.predict(x_validation),
                beta=2,
                zero_division=0,
            )
        )
        best_params = {}
        models_trained = 1
    elif strategy == "grid":
        parameters = {
            "n_estimators": [100, 250],
            "max_depth": [3, 6],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        search = GridSearchCV(
            estimator,
            parameters,
            scoring=F2_SCORER,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        search.fit(x_train, y_train)
        model = search.best_estimator_
        best_score = float(search.best_score_)
        best_params = search.best_params_
        cv_results = pd.DataFrame(search.cv_results_)
        results_path = output_dir / "grid_cv_results.csv"
        heatmap_path = output_dir / "grid_heatmap.png"
        cv_results.to_csv(results_path, index=False)
        _save_grid_heatmap(cv_results, heatmap_path)
        artifact_paths.extend([results_path, heatmap_path])
        models_trained = len(cv_results) * 5
    elif strategy == "random":
        count = trials or 50
        parameters = {
            "n_estimators": randint(80, 501),
            "max_depth": randint(2, 11),
            "learning_rate": loguniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": randint(1, 11),
        }
        search = RandomizedSearchCV(
            estimator,
            parameters,
            n_iter=count,
            scoring=F2_SCORER,
            cv=cv,
            random_state=random_seed,
            n_jobs=-1,
            return_train_score=True,
        )
        search.fit(x_train, y_train)
        model = search.best_estimator_
        best_score = float(search.best_score_)
        best_params = search.best_params_
        cv_results = pd.DataFrame(search.cv_results_)
        results_path = output_dir / "random_cv_results.csv"
        parallel_path = output_dir / "random_parallel_coordinates.png"
        cv_results.to_csv(results_path, index=False)
        _save_parallel_coordinates(cv_results, parallel_path)
        artifact_paths.extend([results_path, parallel_path])
        models_trained = len(cv_results) * 5
    elif strategy == "bayesian":
        count = trials or 100

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            fold_scores: list[float] = []
            for fold_index, (fit_index, score_index) in enumerate(cv.split(x_train, y_train)):
                fold_model = build_estimator(random_seed, **params)
                fold_model.fit(x_train.iloc[fit_index], y_train.iloc[fit_index])
                predictions = fold_model.predict(x_train.iloc[score_index])
                fold_scores.append(
                    fbeta_score(
                        y_train.iloc[score_index],
                        predictions,
                        beta=2,
                        zero_division=0,
                    )
                )
                trial.report(1.0 - float(np.mean(fold_scores)), step=fold_index)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return 1.0 - float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        )
        study.optimize(objective, n_trials=count)
        best_params = study.best_params
        best_score = 1.0 - float(study.best_value)
        model = build_estimator(random_seed, **best_params)
        model.fit(x_train, y_train)
        study_path = output_dir / "bayesian_trials.csv"
        study.trials_dataframe().to_csv(study_path, index=False)
        artifact_paths.extend([study_path, *_save_optuna_plots(study, output_dir)])
        measured_fits = []
        for trial in study.trials:
            if trial.state in {
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
            }:
                measured_fits.append((trial.last_step if trial.last_step is not None else -1) + 1)
        models_trained = sum(measured_fits) + 1
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    elapsed = time.perf_counter() - started
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    threshold = select_threshold(y_validation, validation_probabilities)
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_probabilities, threshold)
    test_predictions = (test_probabilities >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, test_predictions)
    plt.title(f"{strategy.title()} held-out confusion matrix")
    plt.tight_layout()
    confusion_path = output_dir / f"{strategy}_confusion_matrix.png"
    plt.savefig(confusion_path, dpi=160)
    plt.close()

    model_path = output_dir / f"{strategy}_best_model.joblib"
    joblib.dump(model, model_path)
    summary_path = output_dir / f"{strategy}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "strategy": strategy,
                "best_cv_f2": best_score,
                "elapsed_seconds": elapsed,
                "models_trained": models_trained,
                "best_params": best_params,
                "test_metrics": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact_paths.extend([model_path, summary_path, confusion_path])

    mlflow_run_id = None
    with tracked_run(
        f"module-01-{strategy}",
        enabled=mlflow_enabled,
        tags={"team": "smartfinance", "data_version": "give-me-some-credit-v1"},
    ) as run:
        if mlflow_enabled:
            log_run(
                parameters={"strategy": strategy, **best_params},
                metrics={
                    "best_cv_f2": best_score,
                    "elapsed_seconds": elapsed,
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                },
                artifact_paths=artifact_paths,
                model=model,
            )
            mlflow_run_id = run.info.run_id

    return TuningResult(
        strategy=strategy,
        best_score=best_score,
        elapsed_seconds=elapsed,
        models_trained=models_trained,
        best_params=best_params,
        threshold=threshold,
        test_metrics=test_metrics,
        model_path=model_path,
        mlflow_run_id=mlflow_run_id,
    )
