FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    gnupg \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# --- dev target ---
FROM base AS dev
COPY pyproject.toml .
RUN python -m pip install -e ".[dev,vllm]"
COPY . .
RUN chmod +x /app/docker-entrypoint.sh

# --- production target ---
FROM base AS production
COPY pyproject.toml .
RUN python -m pip install ".[vllm]"
COPY . .
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "ocabra.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
