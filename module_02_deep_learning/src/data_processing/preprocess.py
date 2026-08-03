"""Deterministic vocabulary, splits, and batches for financial text."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

LABELS = {"negative": 0, "neutral": 1, "positive": 2}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class Vocabulary:
    def __init__(self, texts: list[str], minimum_frequency: int = 2) -> None:
        counts = Counter(token for text in texts for token in tokenize(text))
        words = sorted(word for word, count in counts.items() if count >= minimum_frequency)
        self.index_to_token = ["<pad>", "<unk>", *words]
        self.token_to_index = {token: index for index, token in enumerate(self.index_to_token)}

    def encode(self, text: str, max_length: int) -> list[int]:
        unknown = self.token_to_index["<unk>"]
        values = [self.token_to_index.get(token, unknown) for token in tokenize(text)]
        return values[:max_length] or [unknown]

    def __len__(self) -> int:
        return len(self.index_to_token)


class FinancialTextDataset(Dataset):
    def __init__(
        self, frame: pd.DataFrame, vocabulary: Vocabulary, max_length: int = 160
    ) -> None:
        self.texts = frame["Sentence"].astype(str).tolist()
        self.labels = [LABELS[label] for label in frame["Sentiment"].str.lower()]
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[list[int], int, str]:
        return (
            self.vocabulary.encode(self.texts[index], self.max_length),
            self.labels[index],
            self.texts[index],
        )


def collate_batch(
    batch: list[tuple[list[int], int, str]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    lengths = torch.tensor([len(tokens) for tokens, _, _ in batch], dtype=torch.long)
    width = max(4, int(lengths.max()))
    tokens = torch.zeros((len(batch), width), dtype=torch.long)
    for row, (values, _, _) in enumerate(batch):
        tokens[row, : len(values)] = torch.tensor(values)
    labels = torch.tensor([label for _, label, _ in batch], dtype=torch.long)
    return tokens, lengths, labels, [text for _, _, text in batch]


def prepare_splits(path: Path, random_seed: int = 42) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path).dropna(subset=["Sentence", "Sentiment"])
    if not set(frame["Sentiment"].str.lower()).issubset(LABELS):
        raise ValueError("Unexpected sentiment label")
    train, remainder = train_test_split(
        frame,
        test_size=0.30,
        stratify=frame["Sentiment"],
        random_state=random_seed,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        stratify=remainder["Sentiment"],
        random_state=random_seed,
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def build_loaders(
    splits: dict[str, pd.DataFrame], batch_size: int = 64
) -> tuple[Vocabulary, dict[str, DataLoader]]:
    vocabulary = Vocabulary(splits["train"]["Sentence"].astype(str).tolist())
    loaders = {
        name: DataLoader(
            FinancialTextDataset(frame, vocabulary),
            batch_size=batch_size,
            shuffle=name == "train",
            collate_fn=collate_batch,
        )
        for name, frame in splits.items()
    }
    return vocabulary, loaders


def load_glove(path: Path | None, vocabulary: Vocabulary, dimension: int = 100) -> torch.Tensor:
    generator = torch.Generator().manual_seed(42)
    matrix = torch.empty((len(vocabulary), dimension)).normal_(0, 0.05, generator=generator)
    matrix[0].zero_()
    if path is None or not path.exists():
        return matrix
    wanted = set(vocabulary.token_to_index)
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            if len(parts) != dimension + 1 or parts[0] not in wanted:
                continue
            matrix[vocabulary.token_to_index[parts[0]]] = torch.tensor(
                np.asarray(parts[1:], dtype=np.float32)
            )
    return matrix
