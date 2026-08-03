"""Required SHAP artifacts and applicant-facing reason generation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def positive_class_values(values):
    if isinstance(values, list):
        return values[-1]
    array = np.asarray(values)
    return array[:, :, -1] if array.ndim == 3 else array


def natural_language_explanation(
    contributions: dict[str, float],
    feature_values: dict[str, float],
    denied: bool,
    top_k: int = 3,
) -> str:
    ordered = sorted(contributions, key=lambda name: abs(contributions[name]), reverse=True)
    adverse = [
        name
        for name in ordered
        if (contributions[name] > 0 and denied) or (contributions[name] < 0 and not denied)
    ][:top_k]
    supportive = [name for name in ordered if name not in adverse][:2]
    decision = "declined" if denied else "approved"
    parts = [f"Your application was {decision}."]
    if adverse:
        reasons = ", ".join(
            f"{name.replace('_', ' ')} ({feature_values.get(name, 0):.2f})"
            for name in adverse
        )
        parts.append(f"The strongest factors affecting that result were {reasons}.")
    if supportive:
        parts.append(
            "Factors supporting the opposite outcome included "
            + ", ".join(name.replace("_", " ") for name in supportive)
            + "."
        )
    parts.append("A credit specialist can review these factors and correct inaccurate data.")
    return " ".join(parts)


def generate_shap_suite(model_path: Path, test_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_path)
    frame = pd.read_csv(test_path)
    x_test = frame.drop(columns=["target_default"])
    sample = x_test.sample(min(1000, len(x_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(sample)
    values = positive_class_values(raw_values)
    base = explainer.expected_value
    if isinstance(base, list | np.ndarray):
        base = np.asarray(base).reshape(-1)[-1]
    explanation = shap.Explanation(
        values=values,
        base_values=np.repeat(float(base), len(sample)),
        data=sample.values,
        feature_names=sample.columns.tolist(),
    )
    shap.plots.bar(explanation, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "global_feature_importance.png", dpi=160)
    plt.close()
    shap.plots.beeswarm(explanation, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "beeswarm.png", dpi=160)
    plt.close()

    probabilities = model.predict_proba(sample)[:, 1]
    denied = np.argsort(probabilities)[-2:].tolist()
    approved = np.argsort(probabilities)[:2].tolist()
    borderline = [int(np.argmin(np.abs(probabilities - 0.5)))]
    selected = approved + denied + borderline
    explanations = []
    for position, index in enumerate(selected, 1):
        shap.plots.waterfall(explanation[index], show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"waterfall_{position}.png", dpi=160)
        plt.close()
        contributions = dict(zip(sample.columns, values[index], strict=True))
        feature_values = sample.iloc[index].to_dict()
        explanations.append(
            {
                "sample_index": int(sample.index[index]),
                "default_probability": float(probabilities[index]),
                "text": natural_language_explanation(
                    contributions, feature_values, probabilities[index] >= 0.5
                ),
            }
        )

    denied_index = denied[-1]
    shap.decision_plot(
        float(base),
        values[denied_index],
        sample.iloc[denied_index],
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "denied_decision_plot.png", dpi=160)
    plt.close()

    interaction = np.asarray(explainer.shap_interaction_values(sample.iloc[:200]))
    if interaction.ndim == 4:
        interaction = interaction[:, :, :, -1]
    mean_interaction = np.abs(interaction).mean(axis=0)
    np.fill_diagonal(mean_interaction, 0)
    upper = np.triu_indices_from(mean_interaction, k=1)
    strongest = np.argsort(mean_interaction[upper])[::-1][:3]
    pairs = [
        {
            "feature_1": sample.columns[upper[0][index]],
            "feature_2": sample.columns[upper[1][index]],
            "mean_absolute_interaction": float(mean_interaction[upper][index]),
        }
        for index in strongest
    ]
    result = {"individual_explanations": explanations, "top_interactions": pairs}
    (output_dir / "shap_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
