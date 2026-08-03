"""Command-line entry point for Module 1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .data import load_split, prepare_and_save
from .tuning import tune


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmartFinance credit-risk optimization")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Create leakage-safe data splits")
    prepare.add_argument(
        "--input",
        type=Path,
        default=Path(os.getenv("DATA_DIR", "data")) / "raw/market_data/Give_some_credit.csv",
    )
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("DATA_DIR", "data")) / "processed/splits",
    )

    tuning = subparsers.add_parser("tune", help="Run one optimization strategy")
    tuning.add_argument(
        "--strategy",
        choices=["baseline", "grid", "random", "bayesian"],
        required=True,
    )
    tuning.add_argument("--trials", type=int)
    tuning.add_argument(
        "--split-dir",
        type=Path,
        default=Path(os.getenv("DATA_DIR", "data")) / "processed/splits",
    )
    tuning.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("ARTIFACT_DIR", "artifacts/module_01")),
    )
    tuning.add_argument("--no-mlflow", action="store_true")
    run_all = subparsers.add_parser("run-all", help="Run the complete required comparison")
    run_all.add_argument(
        "--split-dir",
        type=Path,
        default=Path(os.getenv("DATA_DIR", "data")) / "processed/splits",
    )
    run_all.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("ARTIFACT_DIR", "artifacts/module_01")),
    )
    run_all.add_argument("--no-mlflow", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        paths = prepare_and_save(args.input, args.output_dir)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return

    train = load_split(args.split_dir / "credit_train.csv")
    validation = load_split(args.split_dir / "credit_validation.csv")
    test = load_split(args.split_dir / "credit_test.csv")
    if args.command == "run-all":
        results = []
        for strategy, trials in [
            ("baseline", None),
            ("grid", None),
            ("random", 50),
            ("bayesian", 100),
        ]:
            result = tune(
                strategy=strategy,
                x_train=train[0],
                y_train=train[1],
                x_validation=validation[0],
                y_validation=validation[1],
                x_test=test[0],
                y_test=test[1],
                output_dir=args.output_dir / strategy,
                trials=trials,
                mlflow_enabled=not args.no_mlflow,
            )
            results.append(
                {
                    "strategy": strategy,
                    "best_score": result.best_score,
                    "elapsed_seconds": result.elapsed_seconds,
                    "models_trained": result.models_trained,
                    **result.test_metrics,
                    "mlflow_run_id": result.mlflow_run_id,
                }
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = args.output_dir / "strategy_comparison.csv"
        pd.DataFrame(results).to_csv(comparison_path, index=False)
        print(json.dumps({"comparison": str(comparison_path), "results": results}, indent=2))
        return
    result = tune(
        strategy=args.strategy,
        x_train=train[0],
        y_train=train[1],
        x_validation=validation[0],
        y_validation=validation[1],
        x_test=test[0],
        y_test=test[1],
        output_dir=args.output_dir / args.strategy,
        trials=args.trials,
        mlflow_enabled=not args.no_mlflow,
    )
    print(
        json.dumps(
            {
                "strategy": result.strategy,
                "best_cv_f2": result.best_score,
                "test_metrics": result.test_metrics,
                "model_path": str(result.model_path),
                "mlflow_run_id": result.mlflow_run_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
