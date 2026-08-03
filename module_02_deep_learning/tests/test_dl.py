import torch

from src.model_training.models import (
    AttentionBiLSTM,
    BahdanauSelfAttention,
    TextCNN,
)


def test_cnn_shape() -> None:
    model = TextCNN(vocab_size=50, embedding_dim=16, classes=3, filters=8)
    assert model(torch.randint(0, 50, (4, 12))).shape == (4, 3)


def test_attention_masks_padding() -> None:
    attention = BahdanauSelfAttention(8, 4)
    sequence = torch.randn(2, 5, 8)
    mask = torch.tensor([[True, True, False, False, False], [True] * 5])
    _, weights = attention(sequence, mask)
    assert torch.allclose(weights[0, 2:], torch.zeros(3))
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))


def test_attention_classifier_returns_token_weights() -> None:
    model = AttentionBiLSTM(30, 12, 3, hidden_size=8, layers=1)
    tokens = torch.randint(1, 30, (2, 7))
    logits, weights = model(tokens, torch.tensor([7, 4]), return_attention=True)
    assert logits.shape == (2, 3)
    assert weights.shape == (2, 7)
    assert torch.allclose(weights[1, 4:], torch.zeros(3))
