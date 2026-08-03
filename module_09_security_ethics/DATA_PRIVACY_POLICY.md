# SmartFinance Data Privacy Policy

SmartFinance processes credit-application features, financial documents, market
data, and analyst questions only for risk modelling, research retrieval, model
evaluation, and service security. Production deployments must not ingest direct
identifiers unless a documented lawful basis and business requirement exist.

Direct identifiers detected at ingestion are replaced with typed placeholders
before text reaches a model, vector store, cache, or log. Secrets are supplied
through runtime environment variables. Access to raw credit data, MLflow
artifacts, Airflow logs, and research sessions must use least-privilege roles
and auditable authentication. Data must be encrypted in transit and at rest in
the production environment.

Raw application data is retained only for the period established by the
company's legal and credit-risk retention schedule. Derived training datasets,
embeddings, model artifacts, and logs receive separate retention periods and
deletion procedures. Redis caches expire after one hour by default. Backups and
downstream copies are included in deletion workflows.

Individuals may request access, correction, restriction, portability, or
deletion where applicable, and may request meaningful information about
automated credit decisions. A human reviewer must be available for contested
or borderline decisions. Requests are identity-verified, logged, and answered
within the governing jurisdiction's deadline.

Dataset provenance, purpose limitation, consent or other lawful basis,
cross-border transfers, processors, incidents, and retention exceptions are
reviewed by Legal and Security before production use. Privacy-impact and
fairness assessments are repeated for material data, feature, model, or policy
changes.
