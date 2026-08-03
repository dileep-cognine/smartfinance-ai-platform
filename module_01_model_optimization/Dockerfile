FROM python:3.11.9-slim-bookworm AS builder

WORKDIR /build
COPY module_01_model_optimization/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11.9-slim-bookworm AS runtime

WORKDIR /app/module_01_model_optimization
COPY --from=builder /install /usr/local
COPY module_01_model_optimization/ .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/module_01_model_optimization \
    DATA_DIR=/app/data \
    ARTIFACT_DIR=/app/artifacts/module_01

RUN adduser --disabled-password --gecos "" --uid 10001 appuser \
    && mkdir -p /app/artifacts/module_01 \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD ["python", "-c", "from src.main import build_parser; build_parser(); print('ok')"]

CMD ["python", "-m", "src.main", "--help"]

FROM runtime AS test

USER root
COPY module_01_model_optimization/requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt
USER appuser
CMD ["pytest", "-v"]
