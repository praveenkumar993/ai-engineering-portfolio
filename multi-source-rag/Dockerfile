# Dockerfile — packages the app into a portable, reproducible image.
#
# Why python:3.11-slim:
#   "slim" images have a much smaller footprint than the full python image,
#   which means faster builds, faster deploys, and a smaller attack surface.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]