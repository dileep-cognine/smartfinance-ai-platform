# Module 5 Report - Embeddings

`python -m src.experiments` samples one deterministic 200-document corpus and
compares MiniLM, MPNet, and BGE-small (`BAAI/bge-small-en-v1.5`) via Hugging Face. It writes
all vectors, three UMAP plots, latency, dimension, projected float32 storage for
50,000 documents, ten-query precision@5 using category labels as a reproducible
proxy, duplicate pairs at cosine 0.92, K-Means k=8 and HDBSCAN metrics, and
TF-IDF topic keywords.

Category equality is only a proxy for relevance. The required manual
precision@5 review must inspect the top five texts for each of ten queries and
record relevance independently. UMAP visual separation is qualitative and must
not be used as proof that high-dimensional clusters are valid.

Select the production encoder using retrieval precision first, then latency and
storage. MiniLM is the likely operational baseline because its 384-dimensional
vectors are smaller and faster; that remains a hypothesis until the generated
comparison is reviewed.

The API retains a TF-IDF cold-start fallback so health checks work without a
downloaded model. After choosing a model, persist its dense corpus vectors and
set it as the API index. Base image: `python:3.11.9-slim-bookworm`; record the
measured image size.
