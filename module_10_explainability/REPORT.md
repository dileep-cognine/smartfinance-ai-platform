# Module 10 Report - Advanced Model Explainability

## Implemented Evidence

The SHAP command loads the Module 1 XGBoost artifact and held-out test split. It
generates global importance, beeswarm, two approved/two denied/one borderline
waterfalls, a denied decision plot, three strongest interaction pairs, and
plain-language explanations with an appeal statement. The LIME command creates
word-level HTML explanations for exactly five FinBERT sentences. The Captum
utility calculates 50-step Layer Integrated Gradients through the Module 2
embedding layer and produces signed token attributions.

## Method Comparison

SHAP provides the strongest tabular explanation for FinVision's tree-based
credit model. TreeExplainer is model-aware and, under its assumptions, assigns
additive contributions that reconcile a prediction with a baseline. Global
aggregations are useful for model debugging while local waterfall and decision
plots support individual review. It is relatively stable for a fixed model and
background choice, although correlated credit features can divide credit among
each other and be misread as causal. TreeSHAP is efficient compared with
model-agnostic perturbation, making it the recommended internal and applicant
reason-code foundation after legal review of wording.

LIME is model-agnostic and therefore convenient for FinBERT: it perturbs words
and fits a sparse local surrogate. Its output is intuitive, but fidelity holds
only near the sampled text representation. Token removal can create unnatural
financial sentences, and explanations can change with sampling seed, kernel
width, or correlated terms. One thousand samples per case makes it more
expensive than reading attention but still practical for selected audits.
FinVision should use it as an investigation tool, not the sole regulatory
explanation.

Attention exposes which hidden states received weight when constructing the
BiLSTM context. It is effectively free during inference and visually useful,
but high attention is not proof that changing that token changes the output.
Weights may redistribute without materially changing the prediction.
Consequently attention is best treated as a model diagnostic, not a faithful
attribution or customer-facing reason.

Integrated Gradients accumulates gradients from a padding baseline to the real
embedding input. Under differentiability assumptions it satisfies sensitivity
and completeness more directly than attention and supplies signed token
effects. Results still depend on the baseline and integration path, and 50
forward/backward steps make it substantially more expensive. For deep-model
validation it is the preferred fidelity check against attention and LIME.

FinVision should use TreeSHAP for credit decisions, Integrated Gradients plus
LIME for independent text-model audits, and attention only as supporting
diagnostic evidence. Regulatory output must include model/data version,
prediction date, reason-code definitions, uncertainty and human review, while
avoiding causal claims. Stability should be measured across seeds, background
sets and paraphrases. Disagreements among LIME, attention and Integrated
Gradients are findings to investigate rather than explanations to average away.

## Required Review

Run the tools on the final champion models. Compare the five LIME, attention
and Integrated Gradients token rankings, identify three misleading cases, and
write case-specific observations. Record runtime and stability across repeated
LIME seeds; these findings cannot be determined before model artifacts exist.
