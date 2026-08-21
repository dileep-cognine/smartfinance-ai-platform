FROM python:3.11.9-slim-bookworm AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11.9-slim-bookworm AS runtime
WORKDIR /app/module_07_langgraph_crewai
COPY --from=builder /install /usr/local
COPY module_07_langgraph_crewai/ .
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/module_07_langgraph_crewai
RUN adduser --disabled-password --gecos "" --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD ["python", "-c", "import src.main; print('ok')"]
CMD ["python", "-m", "src.main"]

FROM runtime AS test
CMD ["pytest", "-v"]
