"""CLI for full and parameter-efficient transfer learning."""

from __future__ import annotations

import argparse
from pathlib import Path

from .classification.classifier import train_classifier
from .summarization.summarizer import train_lora_summarizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("finbert", "lora-classifier"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--epochs", type=int, default=5)
    summarize = commands.add_parser("lora-summarizer")
    summarize.add_argument("--pairs", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--epochs", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "finbert":
        train_classifier(args.data, args.output, epochs=args.epochs)
    elif args.command == "lora-classifier":
        train_classifier(
            args.data,
            args.output,
            model_name="facebook/opt-125m",
            use_lora=True,
            epochs=args.epochs,
        )
    else:
        train_lora_summarizer(args.pairs, args.output, epochs=args.epochs)


if __name__ == "__main__":
    main()
