FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install ".[postgres,redis,metrics]"

EXPOSE 8400

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8400/health', timeout=3); exit(0 if r.status_code == 200 else 1)"

CMD ["jaswolf", "serve", "--host", "0.0.0.0", "--port", "8400"]
