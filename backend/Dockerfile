FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS bitnet-builder

ARG BITNET_COMMIT=main
ARG BITNET_PRETUNED_MODEL=bitnet_b1_58-3B
ARG BITNET_KERNEL_FLAVOR=tl2
ARG BITNET_BUILD_JOBS=8
ARG BITNET_ENABLE_CUDA=false
ENV BITNET_COMMIT=${BITNET_COMMIT}
ENV BITNET_PRETUNED_MODEL=${BITNET_PRETUNED_MODEL}
ENV BITNET_KERNEL_FLAVOR=${BITNET_KERNEL_FLAVOR}
ENV BITNET_BUILD_JOBS=${BITNET_BUILD_JOBS}
ENV BITNET_ENABLE_CUDA=${BITNET_ENABLE_CUDA}

WORKDIR /build
COPY scripts/build_bitnet.sh /tmp/build_bitnet.sh
RUN chmod +x /tmp/build_bitnet.sh && /tmp/build_bitnet.sh

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
COPY --from=bitnet-builder /out/bitnet-server /usr/local/bin/bitnet-server
COPY --from=bitnet-builder /out/lib/ /usr/local/lib/bitnet/
ENV LD_LIBRARY_PATH=/usr/local/lib/bitnet:${LD_LIBRARY_PATH}
RUN chmod +x /usr/local/bin/bitnet-server

# --- dev target ---
FROM base AS dev
COPY pyproject.toml .
RUN python -m pip install -e ".[dev,vllm,audio]"
COPY . .
RUN chmod +x /app/docker-entrypoint.sh

# --- production target ---
FROM base AS production
COPY pyproject.toml .
RUN python -m pip install ".[vllm,audio]"
COPY . .
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "ocabra.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
