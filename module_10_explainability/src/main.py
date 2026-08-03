"""CLI for SHAP and LIME explanation generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .integrated_gradients import generate_integrated_gradients
from .lime_explanations import generate_lime_explanations
from .shap_explanations import generate_shap_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    shap_command = commands.add_parser("shap")
    shap_command.add_argument("--model", type=Path, required=True)
    shap_command.add_argument("--test", type=Path, required=True)
    shap_command.add_argument("--output", type=Path, required=True)
    lime_command = commands.add_parser("lime")
    lime_command.add_argument("--model", type=Path, required=True)
    lime_command.add_argument("--sentences", type=Path, required=True)
    lime_command.add_argument("--output", type=Path, required=True)
    ig_command = commands.add_parser("integrated-gradients")
    ig_command.add_argument("--checkpoint", type=Path, required=True)
    ig_command.add_argument("--sentences", type=Path, required=True)
    ig_command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shap":
        generate_shap_suite(args.model, args.test, args.output)
    elif args.command == "lime":
        sentences = json.loads(args.sentences.read_text(encoding="utf-8"))
        if len(sentences) != 5:
            raise ValueError("LIME evaluation requires exactly five sentences")
        generate_lime_explanations(args.model, sentences, args.output)
    else:
        sentences = json.loads(args.sentences.read_text(encoding="utf-8"))
        generate_integrated_gradients(args.checkpoint, sentences, args.output)


if __name__ == "__main__":
    main()
