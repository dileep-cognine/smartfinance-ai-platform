"""Metrics, benchmarks, and visual outputs for text models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader

from ..data_processing.preprocess import Vocabulary, tokenize


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, object]:
    model.eval()
    truth: list[int] = []
    predicted: list[int] = []
    started = time.perf_counter()
    with torch.no_grad():
        for tokens, lengths, labels, _ in loader:
            logits = model(tokens.to(device), lengths.to(device))
            truth.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    elapsed = time.perf_counter() - started
    return {
        "accuracy": sum(a == b for a, b in zip(truth, predicted, strict=True)) / len(truth),
        "f1_macro": f1_score(truth, predicted, average="macro"),
        "inference_ms_per_sample": elapsed * 1000 / len(truth),
        "classification_report": classification_report(
            truth,
            predicted,
            target_names=["negative", "neutral", "positive"],
            output_dict=True,
            zero_division=0,
        ),
    }


def save_history(history: list[dict[str, float]], path: Path) -> None:
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["validation_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="train")
    axes[1].plot(frame["epoch"], frame["validation_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.legend()
        axis.set_xlabel("Epoch")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def attention_explanation(
    model: nn.Module,
    sentence: str,
    vocabulary: Vocabulary,
    device: torch.device,
    output_path: Path | None = None,
) -> list[tuple[str, float]]:
    words = tokenize(sentence)[:160] or ["<unk>"]
    ids = vocabulary.encode(sentence, 160)
    tokens = torch.tensor([ids], device=device)
    lengths = torch.tensor([len(ids)], device=device)
    model.eval()
    with torch.no_grad():
        _, weights = model(tokens, lengths, return_attention=True)
    values = weights[0, : len(words)].cpu().tolist()
    pairs = list(zip(words, values, strict=True))
    if output_path:
        plt.figure(figsize=(max(7, len(words) * 0.45), 2.2))
        sns.heatmap([values], xticklabels=words, yticklabels=["attention"], cmap="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
    return pairs


def save_metrics(metrics: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
