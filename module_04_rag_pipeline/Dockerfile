FROM python:3.11.9-slim-bookworm AS builder
WORKDIR /build
COPY module_04_rag_pipeline/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11.9-slim-bookworm AS runtime
WORKDIR /app/module_04_rag_pipeline
COPY --from=builder /install /usr/local
COPY module_04_rag_pipeline/ .
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DATA_DIR=/app/data \
    PYTHONPATH=/app/module_04_rag_pipeline
RUN adduser --disabled-password --gecos "" --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test
USER root
COPY module_04_rag_pipeline/requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt
USER appuser
CMD ["pytest", "-v"]
