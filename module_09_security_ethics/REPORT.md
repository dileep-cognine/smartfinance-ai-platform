# Module 9 Report - Security and Ethical AI

The API enforces prompt length, detects common direct injections, anonymizes
structured PII, hardens the RAG system prompt, runs a secondary LLM safety
classifier when configured, and filters PII and directive financial advice.
The offline classifier is explicitly labelled as a deterministic fallback.

`src.redteam` contains ten attacks across direct, indirect-document, role-play,
and override categories plus five financial jailbreaks. Run it first against
the raw RAG endpoint and then the guarded endpoint. A human must mark
`attack_succeeded` because refusal quality and information leakage cannot be
reliably inferred from HTTP status. Calculate the before/after pass-rate change
from those reviewed files.

Privacy code combines spaCy NER and regex locations, anonymizes before model
entry, and compares standard versus diffprivlib logistic regression at a
declared epsilon. The privacy-utility discussion must use the measured accuracy
and F1 change.

Fairness code audits age groups, reports demographic parity difference,
equalized odds difference, disparate-impact ratio and group metrics, and
compares unconstrained, ExponentiatedGradient, and Fairlearn GridSearch models.
The dataset has no documented gender field; fabricating one would invalidate
the audit. The model card, privacy policy, and more-than-600-word fintech
Ethical AI Principles are included.

Residual risks include novel/obfuscated injections, unsafe but indirect
financial implications, classifier false negatives, entity-detection errors,
and bias not represented by available protected attributes.

Base image: `python:3.11.9-slim-bookworm`. Record measured image size.
