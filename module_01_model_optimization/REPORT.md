# Module 1 Report - Model Optimization

## Design

The target is highly imbalanced: 10,026 of 150,000 applicants (6.68%) default.
All partitions are therefore stratified. Splitting happens before fitting
imputation and clipping statistics, preventing validation/test information from
leaking into training. Search uses five-fold stratified cross-validation and
F2, which weights recall more heavily than precision. A separate validation
threshold selects maximum default recall while requiring at least 70% precision
when such a threshold exists.

The image uses `python:3.11.9-slim-bookworm`: XGBoost, Optuna, and scikit-learn
run on CPU and do not justify a CUDA runtime. Record the measured final image
size here after building: `docker images smartfinance-model_optimizer`.

## Required Experimental Results

Do not fill this section with estimated values. Run all strategies and copy
their generated JSON/CSV results and MLflow run IDs.

| Strategy | Best CV F2 | Wall time (s) | Models trained | Test precision | Test recall | Test F2 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | TODO | TODO | 1 | TODO | TODO | TODO |
| Grid | TODO | TODO | TODO | TODO | TODO | TODO |
| Random (50) | TODO | TODO | 250 | TODO | TODO | TODO |
| Bayesian (100) | TODO | TODO | measured with pruning | TODO | TODO | TODO |

## Production Recommendation

Subject to the measured comparison confirming acceptable held-out precision and
recall, Bayesian optimization is the recommended weekly search strategy.
Grid Search is useful for a small, deliberately bounded diagnostic experiment
because it evaluates every Cartesian combination and makes parameter
interactions easy to inspect. Its cost grows multiplicatively, however. The
current two-value grid across five parameters contains 32 candidates and
requires 160 fitted models under five-fold validation. Adding only one value to
each dimension would increase that to 1,215 fits. It also wastes computation in
regions that early results already show to be weak and cannot naturally express
continuous search spaces.

Random Search covers a broader space and is a strong baseline. Fifty trials
require 250 cross-validation fits and explore continuous learning rate,
subsample and feature-subsample distributions rather than a few arbitrary
points. It can outperform the compact grid when only a subset of parameters
strongly controls performance. Each random trial is independent, which also
makes it simple to parallelize. Its limitation is that knowledge gained from
completed trials does not guide later samples.

Optuna's TPE sampler uses earlier observations to allocate later trials toward
promising regions. The median pruner evaluates intermediate fold results and
terminates candidates that are unlikely to improve the study, so 100 logical
trials need not complete all 500 possible fold fits. That behavior should
deliver the best compute-to-improvement trade-off for recurring retraining. The
recommendation must be reversed if the generated comparison shows no material
F2 or recall improvement over Random Search after accounting for elapsed time.
In that case, the simpler and more parallel Random Search is operationally
preferable.

The Airflow workflow should validate schema and row counts, create leakage-safe
splits, and start an MLflow parent run tagged with code and data versions. The
tuning task logs every candidate and registers only the winning serialized
pipeline. Evaluation occurs once on the held-out test set, using a threshold
selected on validation data to maximize default recall while maintaining at
least 70 percent precision. The candidate advances through Staging to the
`champion` alias only when its test F1 exceeds the current production model by
more than 0.02 and its precision policy passes. Otherwise, production remains
unchanged. Drift alerts may trigger an additional run, but never automatic
promotion. The weekly schedule, model alias, artifacts, comparison table and
notification give auditors a reproducible record and provide rollback to the
previous champion.
