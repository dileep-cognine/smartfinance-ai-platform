# Module 2 Report - Deep Learning Architectures

## Design

All three models use the same deterministic stratified 70/15/15 Financial
PhraseBank split, vocabulary, 100-dimensional GloVe matrix, class-weighted
cross-entropy, and held-out test set. The CNN applies parallel widths 2, 3, and
4, capturing local financial phrases such as "profit warning" or "revenue
growth". The two-layer bidirectional LSTM uses 128 hidden units per direction
and gradient clipping at norm 1.0. The third model adds Bahdanau self-attention.

For hidden vectors `h_t`, additive attention computes
`e_t = v^T tanh(W h_t + b)`, masks padding, normalizes
`alpha_t = exp(e_t) / sum_j exp(e_j)`, and returns
`c = sum_t alpha_t h_t`. Unlike selecting one terminal LSTM state, the context
can combine evidence from any position and provides inspectable token weights.

## Reproducible Run

Run `python -m src.main --epochs 20 --glove /path/glove.6B.100d.txt`.
The script writes learning curves, per-class precision/recall/F1, inference
milliseconds per sample, model sizes, a comparison CSV, and five attention
heatmaps.

## Measured Findings

Populate from `architecture_comparison.csv`; do not invent values.

| Architecture | Accuracy | F1-macro | ms/sample | Size MB |
|---|---:|---:|---:|---:|
| CNN | pending run | pending run | pending run | pending run |
| BiLSTM | pending run | pending run | pending run | pending run |
| BiLSTM + Attention | pending run | pending run | pending run | pending run |

Longer articles should favor the bidirectional sequence models because they
retain order and evidence across distant tokens; the measured test subsets must
confirm or reject that hypothesis. Attention weights are explanatory signals,
not guaranteed causal attributions, and are compared with Integrated Gradients
in Module 10.

The container uses pinned CPU PyTorch on `python:3.11.9-slim-bookworm` so the
assessment runs on hosts without NVIDIA Container Toolkit. The same code detects
and uses CUDA when deployed in a GPU-enabled image.
