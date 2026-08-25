"""Captum Integrated Gradients for Module 2's attention BiLSTM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from captum.attr import LayerIntegratedGradients

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module_02_deep_learning.src.data_processing.preprocess import tokenize
from module_02_deep_learning.src.model_training.models import AttentionBiLSTM


def explain_sentence(
    model,
    token_ids: list[int],
    words: list[str],
    device: torch.device,
) -> list[float]:
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    baseline = torch.zeros_like(tokens)
    lengths = torch.tensor([len(token_ids)], device=device)

    def forward(input_tokens, input_lengths):
        return model(input_tokens, input_lengths)

    model.eval()
    with torch.no_grad():
        target = int(model(tokens, lengths).argmax(dim=1))
    integrated = LayerIntegratedGradients(forward, model.embedding)
    attributions = integrated.attribute(
        tokens,
        baselines=baseline,
        additional_forward_args=(lengths,),
        target=target,
        n_steps=50,
    )
    if isinstance(attributions, (tuple, list)):
        attr_tensor = attributions[0]
    else:
        attr_tensor = attributions
    assert isinstance(attr_tensor, torch.Tensor)
    scores = attr_tensor.sum(dim=-1).squeeze(0)
    scores = scores / (torch.linalg.vector_norm(scores) + 1e-9)
    return scores[: len(words)].detach().cpu().tolist()


def save_attribution(
    words: list[str], scores: list[float], path: Path, title: str
) -> None:
    plt.figure(figsize=(max(7, len(words) * 0.45), 2.2))
    sns.heatmap(
        [scores],
        xticklabels=words,
        yticklabels=["integrated gradients"],
        center=0,
        cmap="coolwarm",
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def generate_integrated_gradients(
    checkpoint_path: Path,
    sentences: list[str],
    output_dir: Path,
) -> list[dict]:
    if len(sentences) != 5:
        raise ValueError("Integrated Gradients evaluation requires exactly five sentences")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocabulary = checkpoint["vocabulary"]
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    model = AttentionBiLSTM(len(vocabulary), 100, 3)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, sentence in enumerate(sentences, 1):
        words = tokenize(sentence)[:160] or ["<unk>"]
        ids = [token_to_index.get(word, 1) for word in words]
        scores = explain_sentence(model, ids, words, device)
        save_attribution(
            words,
            scores,
            output_dir / f"integrated_gradients_{index}.png",
            f"Example {index}",
        )
        results.append(
            {
                "sentence": sentence,
                "token_attributions": list(zip(words, scores, strict=True)),
            }
        )
    (output_dir / "integrated_gradients.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results
