"""Reusable training loop with clipping and learning history."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    for tokens, lengths, labels, _ in loader:
        tokens, lengths, labels = tokens.to(device), lengths.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(tokens, lengths)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += labels.size(0)
    return total_loss / count, correct / count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        with torch.no_grad():
            validation_loss, validation_accuracy = run_epoch(
                model, validation_loader, criterion, device
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
    if best_state:
        model.load_state_dict(best_state)
    return model, history
