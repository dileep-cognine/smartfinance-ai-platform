import torch
from transformers import OPTConfig, OPTForSequenceClassification

from src.classification.classifier import parameter_counts
from src.main import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_PATH, build_parser


def test_parameter_counts() -> None:
    model = torch.nn.Linear(5, 2)
    counts = parameter_counts(model)
    assert counts["total"] == 12
    assert counts["trainable"] == 12


def test_opt_exposes_required_lora_targets() -> None:
    model = OPTForSequenceClassification(
        OPTConfig(
            vocab_size=100,
            hidden_size=16,
            num_hidden_layers=1,
            ffn_dim=32,
            num_attention_heads=2,
            num_labels=3,
        )
    )
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    assert {"q_proj", "v_proj"}.issubset(names)


def test_run_all_uses_shared_data_and_artifact_paths() -> None:
    args = build_parser().parse_args(["run-all", "--no-mlflow"])
    assert args.data == DEFAULT_DATA_PATH
    assert args.output == DEFAULT_ARTIFACT_DIR
    assert args.no_mlflow
