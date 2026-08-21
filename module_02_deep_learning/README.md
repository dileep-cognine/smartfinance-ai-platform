# Module 2: Deep Learning for Financial Sentiment

This module compares three PyTorch models for classifying Financial PhraseBank
sentences as `negative`, `neutral`, or `positive`:

- `TextCNN`: parallel 1D convolution filters for local phrase patterns.
- `BiLSTMClassifier`: a two-layer bidirectional LSTM for sequence context.
- `AttentionBiLSTM`: the BiLSTM with additive (Bahdanau) attention and
  token-level attention weights.

All models use the same deterministic 70/15/15 stratified split, vocabulary,
class-weighted loss, and held-out test data. The code uses CUDA automatically
when the installed PyTorch build exposes a compatible GPU; the current project
requirements install the CPU PyTorch build for portable execution.

## Data

The default input is the shared project dataset:

```text
../data/processed/cleaned/financial_phrasebank_clean.csv
```

It must contain `Sentence` and `Sentiment` columns. Run commands from this
module directory. Data and generated artifacts are deliberately kept outside
the module folder so other modules can reuse them.

## Run

Activate the project virtual environment from the repository root, then change
to this directory.

```powershell
python -m pytest -q
python -m src.main --epochs 1 --batch-size 32
```

The one-epoch command is a smoke test. Run the full experiment with:

```powershell
python -m src.main --epochs 20 --batch-size 64
```

To use 100-dimensional GloVe vectors when a local file is available:

```powershell
python -m src.main --glove C:\path\to\glove.6B.100d.txt --epochs 20
```

Without `--glove`, the module uses deterministic randomly initialized
100-dimensional embeddings, so it remains runnable without downloading GloVe.

## Outputs

Results are saved to the shared location:

```text
../artifacts/module_02/
├── cnn.pt, bilstm.pt, attention.pt
├── *_learning_curves.png
├── *_metrics.json
├── attention_example_1.png ... attention_example_5.png
├── attention_example_1.json ... attention_example_5.json
└── architecture_comparison.csv
```

Use `architecture_comparison.csv` for the measured accuracy, macro F1,
inference latency, and model-size comparison in `REPORT.md`.

## Quality checks

```powershell
python -m pytest -q
python -m ruff check .
```

The tests cover the CNN output shape, attention padding masks, and the
attention classifier's token-weight output. See `REPORT.md` for design
rationale and analysis guidance.
