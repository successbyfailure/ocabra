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

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS llama-builder

ARG LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp
ARG LLAMA_CPP_REF=master

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} ${LLAMA_CPP_REPO} /build/llama.cpp
RUN cmake -S /build/llama.cpp -B /build/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build /build/llama.cpp/build --target llama-server -j4

FROM docker:28-cli AS docker-cli

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PATH=/usr/local/cuda/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    gnupg \
    curl \
    git \
    ninja-build \
    cuda-toolkit-12-4 \
    sox \
    libsox-fmt-all \
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
COPY --from=docker-cli /usr/local/bin/docker /usr/bin/docker
COPY --from=bitnet-builder /out/bitnet-server /usr/local/bin/bitnet-server
COPY --from=bitnet-builder /out/lib/ /usr/local/lib/bitnet/
COPY --from=llama-builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama-builder /build/llama.cpp/build/bin/lib*.so* /usr/local/lib/llama_cpp/
ENV LD_LIBRARY_PATH=/usr/local/lib/llama_cpp:/usr/local/lib/bitnet:${LD_LIBRARY_PATH}
RUN chmod +x /usr/local/bin/bitnet-server /usr/local/bin/llama-server

# --- dev target ---
FROM base AS dev
ARG SGLANG_VERSION=0.5.9
ARG VLLM_OMNI_VERSION=0.18.0
COPY pyproject.toml .
RUN python -m pip install -e ".[dev,vllm,audio]"
RUN python -m pip install --upgrade "transformers>=5.0"
RUN python -m pip install "qwen-tts>=0.1.1"
RUN python -m venv /opt/sglang-venv \
    && /opt/sglang-venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/sglang-venv/bin/pip install "sglang==${SGLANG_VERSION}"
RUN python -m venv /opt/voxtral-venv \
    && /opt/voxtral-venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/voxtral-venv/bin/pip install "vllm>=0.18.0" "vllm-omni>=${VLLM_OMNI_VERSION}"
RUN python -m venv /opt/chatterbox-venv \
    && /opt/chatterbox-venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/chatterbox-venv/bin/pip install "chatterbox-tts>=0.1.7"
COPY . .
RUN chmod +x /app/docker-entrypoint.sh

# --- production target ---
FROM base AS production
ARG SGLANG_VERSION=0.5.9
ARG VLLM_OMNI_VERSION=0.18.0
COPY pyproject.toml .
RUN python -m pip install ".[vllm,audio]"
RUN python -m pip install --upgrade "transformers>=5.0"
RUN python -m pip install "qwen-tts>=0.1.1"
RUN python -m venv /opt/sglang-venv \
    && /opt/sglang-venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/sglang-venv/bin/pip install "sglang==${SGLANG_VERSION}"
RUN python -m venv /opt/voxtral-venv \
    && /opt/voxtral-venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/voxtral-venv/bin/pip install "vllm>=0.18.0" "vllm-omni>=${VLLM_OMNI_VERSION}"
RUN python -m venv /opt/chatterbox-venv \
    && /opt/chatterbox-venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/chatterbox-venv/bin/pip install "chatterbox-tts>=0.1.7"
COPY . .
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "ocabra.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
