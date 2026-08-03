# FinVision Ethical AI Principles

FinVision uses AI to support financial analysis and credit-risk decisions, not
to remove institutional responsibility. Every deployed system has a named
business owner, model owner, data owner, security owner, and compliance
approver. Accountability remains with these people even when a vendor model,
external API, or automated workflow contributes to the result. Material
incidents, overrides, complaints, and monitoring alerts are recorded and
reviewed by the AI governance committee.

## Transparency

Customers and analysts must know when they are interacting with AI and what role
it plays. Credit applicants receive the principal factors affecting an adverse
decision in plain language. Internal users can inspect the champion model
version, training-data version, evaluation date, intended use, limitations, and
reason-code method. Research answers cite filing names and chunks. Generated
content is distinguished from source text. FinVision does not claim that
attention weights, SHAP values, or language-model explanations prove causality.

## Accountability and Validation

No model enters production only because an aggregate metric improved. An
independent reviewer checks data provenance, leakage, calibration, subgroup
performance, security tests, operational rollback, and documentation.
MLflow records parameters, metrics, code/data tags, and artifacts. Promotion
requires a documented threshold and approval; the production service resolves a
named alias rather than an untracked file. Significant changes trigger a fresh
validation, not an informal update.

## Fairness

FinVision evaluates outcomes across legally and ethically relevant groups using
demographic parity, equalized odds, selection rates, recall, false-positive and
false-negative rates. Aggregate parity can hide harm, so the review includes
intersectional groups where sample sizes permit and records uncertainty when
they do not. Protected attributes are used for controlled auditing, access is
restricted, and they are not silently inferred for decision-making. Mitigation
methods are compared on both utility and fairness. A numerical fairness target
does not replace fair-lending legal review or analysis of whether input
features act as proxies.

## Privacy and Data Governance

Only data necessary for a documented purpose is collected. Direct identifiers
are removed before model, embedding, cache, or log ingestion unless explicitly
required and legally approved. Raw, derived, vector, artifact, and backup data
have separate retention schedules. Access follows least privilege and is
audited. Encryption is required in transit and at rest. Differential privacy is
evaluated where aggregate learning can tolerate its utility cost. Customers can
request access, correction, restriction, deletion where applicable, and human
review of consequential automated outcomes.

## Human Oversight

Credit models provide risk estimates and reason codes; qualified staff retain
authority and responsibility for consequential decisions. Borderline cases,
low-confidence predictions, out-of-distribution inputs, conflicting evidence,
and customer appeals are routed to humans. The research-agent workflow pauses
after risk assessment and requires explicit approval before report synthesis.
After three rejected automated revisions, LangGraph escalates instead of
continuing indefinitely. Overrides include a reason and are monitored for
patterns that indicate model or policy defects.

## Safety and Security

Retrieved documents and user prompts are treated as untrusted content.
Applications enforce length limits, prompt-injection detection, hardened system
instructions, output classification, PII removal, and financial-advice filters.
Red-team suites cover direct and indirect injection, role-play, instruction
override, and attempts to obtain personalized recommendations. Residual risk is
documented because no prompt defense guarantees safety. Secrets are runtime
variables, containers run as non-root, dependencies are pinned, and services
expose only necessary local ports.

## Reliability and Monitoring

FinVision monitors schema validity, missingness, data and prediction drift,
accuracy, F1, calibration, latency, errors, subgroup performance, and security
events. Threshold breaches open an investigation and may trigger retraining;
they do not automatically approve a model. Airflow tasks have retries and
service-level deadlines. Every deployment has a rollback path to a previously
validated alias. Providers, models, datasets, and prompts are versioned so an
incident can be reproduced.

## Governance and Continuous Improvement

The governance committee meets regularly and includes engineering, risk,
security, privacy, legal, compliance, product, and customer-support
representation. It approves use cases according to impact, reviews monitoring
and complaints, and can suspend a system. Vendor claims are independently
tested. Policies are updated when regulation, products, populations, or threat
models change. FinVision publishes internally what failed as well as what
succeeded, because honest limitations and corrective actions are necessary for
trustworthy financial AI.
