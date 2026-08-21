"""CLI for full and parameter-efficient transfer learning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .classification.classifier import train_classifier
from .summarization.summarizer import train_lora_summarizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))) / (
    "processed/cleaned/financial_phrasebank_clean.csv"
)
DEFAULT_ARTIFACT_DIR = Path(
    os.getenv("ARTIFACT_DIR", str(PROJECT_ROOT / "artifacts/module_03"))
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("finbert", "lora-classifier"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
        command.add_argument(
            "--output", type=Path, default=DEFAULT_ARTIFACT_DIR / name.replace("-", "_")
        )
        command.add_argument("--epochs", type=int, default=5)
        command.add_argument("--no-mlflow", action="store_true")
    run_all = commands.add_parser(
        "run-all", help="Run FinBERT and LoRA classification sequentially"
    )
    run_all.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    run_all.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_DIR)
    run_all.add_argument("--epochs", type=int, default=5)
    run_all.add_argument("--no-mlflow", action="store_true")
    summarize = commands.add_parser("lora-summarizer")
    summarize.add_argument("--pairs", type=Path, required=True)
    summarize.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_DIR / "lora_summarizer")
    summarize.add_argument("--epochs", type=int, default=3)
    summarize.add_argument("--no-mlflow", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "finbert":
        train_classifier(
            args.data,
            args.output,
            epochs=args.epochs,
            mlflow_enabled=not args.no_mlflow,
        )
    elif args.command == "lora-classifier":
        train_classifier(
            args.data,
            args.output,
            model_name="facebook/opt-125m",
            use_lora=True,
            epochs=args.epochs,
            mlflow_enabled=not args.no_mlflow,
        )
    elif args.command == "run-all":
        results = {
            "finbert": train_classifier(
                args.data,
                args.output / "finbert",
                epochs=args.epochs,
                mlflow_enabled=not args.no_mlflow,
            ),
            "lora_classifier": train_classifier(
                args.data,
                args.output / "lora_classifier",
                model_name="facebook/opt-125m",
                use_lora=True,
                epochs=args.epochs,
                mlflow_enabled=not args.no_mlflow,
            ),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "classification_comparison.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    else:
        train_lora_summarizer(
            args.pairs,
            args.output,
            epochs=args.epochs,
            mlflow_enabled=not args.no_mlflow,
        )


if __name__ == "__main__":
    main()
