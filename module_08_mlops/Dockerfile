FROM python:3.11.9-slim-bookworm AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11.9-slim-bookworm AS runtime
WORKDIR /app/module_08_mlops
COPY --from=builder /install /usr/local
COPY module_08_mlops/ .
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DATA_DIR=/app/data \
    PYTHONPATH=/app/module_08_mlops
RUN adduser --disabled-password --gecos "" --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test
CMD ["pytest", "-v"]
