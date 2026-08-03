# Model Card: SmartFinance Credit-Risk Classifier

## Model Details

XGBoost binary classifier optimized with five-fold stratified F2 search and
tracked in MLflow. The production version must be resolved through the
`smartfinance-credit-risk@champion` alias.

## Intended Use

Decision support for trained credit-risk staff evaluating default likelihood.
Outputs may prioritize applications for review and provide reason codes.

## Out-of-Scope Uses

The model must not autonomously deny credit, set prices, infer protected
attributes, determine criminality or employability, or be applied outside the
population and product represented by the training data.

## Data

The assessment uses Give Me Some Credit. It contains severe class imbalance and
does not contain a documented gender field. Age is evaluated in coarse groups.
Dataset provenance, license, sampling period, geographic relevance, and feature
definitions must be confirmed before production use.

## Evaluation

Report held-out precision, recall, F1, F2, ROC-AUC, calibration, confusion
matrix, and metrics by age group. Populate measured values from the final
MLflow champion run. The business policy requires maximizing default recall
while maintaining at least 70% precision.

## Ethical Considerations

Age-related disparities may reflect structural inequity or dataset selection.
Fairness mitigation is evaluated alongside accuracy, but satisfying one group
metric does not establish lawful or fair lending. Applicants require clear
reason codes, a human appeal path, and correction procedures.

## Limitations and Monitoring

Historical labels may contain policy bias. Distribution shifts, missing-value
patterns, economic regime changes, calibration, and group metrics require
ongoing monitoring. PSI above 0.2 triggers investigation/retraining, not
automatic production promotion.
