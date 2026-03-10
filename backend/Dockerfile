FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# --- dev target ---
FROM base AS dev
COPY pyproject.toml .
RUN pip install -e ".[dev]"
COPY . .

# --- production target ---
FROM base AS production
COPY pyproject.toml .
RUN pip install .
COPY . .

EXPOSE 8000
CMD ["uvicorn", "ocabra.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
