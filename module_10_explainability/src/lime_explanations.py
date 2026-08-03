"""LIME word-level explanations for a saved FinBERT classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASS_NAMES = ["negative", "neutral", "positive"]


def generate_lime_explanations(
    model_path: Path,
    sentences: list[str],
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def predict(texts: list[str]) -> np.ndarray:
        batch = tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            return torch.softmax(model(**batch).logits, dim=1).cpu().numpy()

    explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=42)
    results = []
    for index, sentence in enumerate(sentences, 1):
        probabilities = predict([sentence])[0]
        label = int(np.argmax(probabilities))
        explanation = explainer.explain_instance(
            sentence, predict, labels=[label], num_features=12, num_samples=1000
        )
        weights = explanation.as_list(label=label)
        explanation.save_to_file(str(output_dir / f"lime_{index}.html"))
        results.append(
            {
                "sentence": sentence,
                "predicted_label": CLASS_NAMES[label],
                "probability": float(probabilities[label]),
                "word_weights": weights,
            }
        )
    (output_dir / "lime_explanations.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results
