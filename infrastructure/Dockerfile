# syntax=docker/dockerfile:1.6
# Multi-stage build for the Airline Churn API.
#   Stage 1: install dependencies in a clean venv
#   Stage 2: copy only what's needed at runtime
FROM python:3.11-slim AS builder
WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


FROM python:3.11-slim AS runtime
WORKDIR /app

# Create non-root user
RUN groupadd --system app && useradd --system --gid app --create-home app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only what's needed at runtime
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY configs/ /app/configs/

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# Note: src/api/app.py exposes `app`, so the module path is src.api.app:app
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
