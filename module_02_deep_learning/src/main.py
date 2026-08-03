"""Train and compare all Module 2 architectures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch

from .data_processing.preprocess import LABELS, build_loaders, load_glove, prepare_splits
from .evaluation.evaluate import (
    attention_explanation,
    evaluate_model,
    save_history,
    save_metrics,
)
from .model_training.models import AttentionBiLSTM, BiLSTMClassifier, TextCNN
from .model_training.train import train_model

EXAMPLES = [
    "The company reported strong revenue growth and higher profit.",
    "Shares fell after management warned about weaker demand.",
    "The board announced a merger subject to regulatory approval.",
    "Interest rates remained unchanged after the central bank meeting.",
    "Operating income improved but debt increased substantially.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.getenv("DATA_DIR", "data"))
        / "processed/cleaned/financial_phrasebank_clean.csv",
    )
    parser.add_argument("--glove", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/module_02"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(42)
    splits = prepare_splits(args.data)
    vocabulary, loaders = build_loaders(splits, args.batch_size)
    embeddings = load_glove(args.glove, vocabulary)
    counts = splits["train"]["Sentiment"].str.lower().value_counts()
    weights = torch.tensor(
        [len(splits["train"]) / (len(LABELS) * counts[label]) for label in LABELS],
        dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    factories = {
        "cnn": lambda: TextCNN(len(vocabulary), 100, 3, embeddings),
        "bilstm": lambda: BiLSTMClassifier(len(vocabulary), 100, 3, embeddings),
        "attention": lambda: AttentionBiLSTM(len(vocabulary), 100, 3, embeddings),
    }
    comparison = []
    trained = {}
    for name, factory in factories.items():
        model, history = train_model(
            factory(), loaders["train"], loaders["validation"], weights, device, args.epochs
        )
        metrics = evaluate_model(model, loaders["test"], device)
        model_path = args.output / f"{name}.pt"
        torch.save(
            {"state_dict": model.state_dict(), "vocabulary": vocabulary.index_to_token},
            model_path,
        )
        metrics["model_size_mb"] = model_path.stat().st_size / (1024 * 1024)
        save_history(history, args.output / f"{name}_learning_curves.png")
        save_metrics(metrics, args.output / f"{name}_metrics.json")
        summary_metrics = {
            key: value
            for key, value in metrics.items()
            if key != "classification_report"
        }
        comparison.append({"architecture": name, **summary_metrics})
        trained[name] = model
    for index, sentence in enumerate(EXAMPLES, 1):
        pairs = attention_explanation(
            trained["attention"],
            sentence,
            vocabulary,
            device,
            args.output / f"attention_example_{index}.png",
        )
        (args.output / f"attention_example_{index}.json").write_text(
            json.dumps(pairs, indent=2), encoding="utf-8"
        )
    pd.DataFrame(comparison).to_csv(args.output / "architecture_comparison.csv", index=False)


if __name__ == "__main__":
    main()
