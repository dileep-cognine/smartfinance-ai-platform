# Module 10: Advanced Model Explainability (XAI)

This module provides regulatory-grade interpretability across credit risk models and deep NLP sentiment classifiers:
1. **SHAP (TreeExplainer)**: Global feature importance, beeswarm distributions, local applicant waterfall plots, cumulative decision trajectories, top interaction pairs, and automated plain-language adverse action reasons.
2. **LIME (Local Interpretable Model-agnostic Explanations)**: Local surrogate word perturbation models generating interactive HTML dashboards for FinBERT sentiment classification.
3. **Captum (Layer Integrated Gradients)**: 50-step path-integrated token-level signed attributions through BiLSTM embedding layers.

---

## 1. Interpretability Method Comparison

| Method | Target Model | Scope | Computation Cost | Regulatory Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **TreeSHAP** | XGBoost Credit Model | Global & Local | Fast ($O(TLD^2)$) | Primary adverse action notice generator (FCRA/ECOA reason codes) |
| **LIME** | FinBERT Classifier | Local Instance | Moderate (1,000 text samples) | Independent validation of text model predictions |
| **Integrated Gradients**| Attention-BiLSTM | Token-level | Higher (50 path steps) | Deep model verification satisfying sensitivity & completeness axioms |

---

## 2. Running & Testing

### 1. Run Automated Unit Tests
From `module_10_explainability`:

```powershell
python -m pytest tests -v
```

### 2. Generate the Complete SHAP Suite
```powershell
python -m src.main shap --model "..\artifacts\module_01\bayesian\bayesian_best_model.joblib" --test "..\data\processed\splits\credit_test.csv" --output "..\artifacts\module_10\shap"
```

> **Generated Artifacts in `artifacts/module_10/shap/`**:
> - `global_feature_importance.png` (Global mean |SHAP| feature bar chart)
> - `beeswarm.png` (Feature value distributions vs. SHAP contributions)
> - `waterfall_1.png` to `waterfall_5.png` (Approved, denied, and borderline applicant decision paths)
> - `denied_decision_plot.png` (Cumulative decision path for denied applicants)
> - `shap_summary.json` (Structured reason codes and applicant explanations)

### 3. Generate LIME Financial Text Explanations
```powershell
# Create sample 5-sentence benchmark
Set-Content -Path "sample_sentences.json" -Value '[
  "Revenue grew by 15 percent exceeding analyst quarterly expectations.",
  "Operating margins declined due to unexpected supply chain disruptions.",
  "The company maintained a stable cash reserve with zero debt maturities.",
  "Severe regulatory fines resulted in significant net losses this quarter.",
  "Management guidance remains cautiously optimistic for the second half."
]'

# Run LIME explainer
python -m src.main lime --model "ProsusAI/finbert" --sentences "sample_sentences.json" --output "..\artifacts\module_10\lime"
```

> **Generated Artifacts in `artifacts/module_10/lime/`**:
> - `lime_1.html` to `lime_5.html` (Interactive HTML dashboards)
> - `lime_explanations.json` (Token weights and class probabilities)

### 4. Generate Captum Layer Integrated Gradients
```powershell
python -m src.main integrated-gradients --checkpoint "..\artifacts\module_02\attention.pt" --sentences "sample_sentences.json" --output "..\artifacts\module_10\captum"
```

> **Generated Artifacts in `artifacts/module_10/captum/`**:
> - `integrated_gradients_1.png` to `integrated_gradients_5.png` (Word attribution heatmaps)
> - `integrated_gradients.json` (Signed attribution scores)
