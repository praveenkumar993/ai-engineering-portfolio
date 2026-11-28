# Dockerfile for ai-code-reviewer
#
# Multi-stage build: builder installs dependencies, final image
# copies only what's needed. This keeps the image small -- we
# don't ship pip, build tools, or the wheel cache to production.
#
# Why Python 3.12-slim?
#   slim strips out documentation, test suites, and other files
#   that aren't needed at runtime. The image is about 60% smaller
#   than the full python:3.12 image with no functional difference
#   for a FastAPI service.

# --- Stage 1: builder ---
FROM python:3.12-slim AS builder

WORKDIR /build

# Copy requirements first so Docker caches the pip install layer.
# If requirements.txt hasn't changed, Docker reuses the cached
# layer even if your code changed -- much faster rebuilds.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# --- Stage 2: final image ---
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY tests/ ./tests/
COPY run.py .

# Create the logs directory the logger writes to
RUN mkdir -p logs

# The service listens on 8000
EXPOSE 8000

# Non-root user for security -- running as root inside a container
# is unnecessary and increases blast radius if the container is
# compromised.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Health check -- Docker monitors this and marks the container
# unhealthy if it fails. Same /health endpoint we built in Step 12.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the FastAPI server
CMD ["python", "run.py"]