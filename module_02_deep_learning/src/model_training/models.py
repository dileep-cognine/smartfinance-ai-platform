"""CNN, BiLSTM, and additive-attention sentiment classifiers."""

from __future__ import annotations

import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        classes: int,
        embedding_weights: torch.Tensor | None = None,
        filters: int = 100,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(embedding_dim, filters, kernel_size=size) for size in (2, 3, 4)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters * 3, classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [
            torch.max(torch.relu(convolution(embedded)), dim=2).values
            for convolution in self.convolutions
        ]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        classes: int,
        embedding_weights: torch.Tensor | None = None,
        hidden_size: int = 128,
        layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, classes)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(self.embedding(tokens))
        return outputs

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(self.embedding(tokens))
        final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(final))


class BahdanauSelfAttention(nn.Module):
    """Additive attention: score(h_t) = v^T tanh(W h_t + b)."""

    def __init__(self, input_size: int, attention_size: int = 128) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, attention_size)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        energies = self.score(torch.tanh(self.projection(sequence))).squeeze(-1)
        energies = energies.masked_fill(~mask, torch.finfo(energies.dtype).min)
        weights = torch.softmax(energies, dim=1)
        context = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)
        return context, weights


class AttentionBiLSTM(BiLSTMClassifier):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        classes: int,
        embedding_weights: torch.Tensor | None = None,
        hidden_size: int = 128,
        layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(
            vocab_size,
            embedding_dim,
            classes,
            embedding_weights,
            hidden_size,
            layers,
            dropout,
        )
        self.attention = BahdanauSelfAttention(hidden_size * 2, hidden_size)

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encode(tokens)
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        context, weights = self.attention(outputs, mask)
        logits = self.classifier(self.dropout(context))
        return (logits, weights) if return_attention else logits
