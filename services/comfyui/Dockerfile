FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ffmpeg \
    gcc \
    libc6-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ComfyUI pide torch reciente con cu130 en las series 20 y posteriores (su
# README, 2026-11). La imagen anterior (pytorch 2.5.1 + cu124) ya no carga los
# nodos de LTX-2.5 ni los pesos int8 «convrot». El driver del hub (615) da CUDA 13.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install torch torchvision torchaudio --index-url ${TORCH_INDEX_URL}

# Pinned (INF-07) to upstream HEAD as of 2026-11-19 (no image of this service
# runs on the hub to copy the commit from). Override with --build-arg.
ARG COMFYUI_REF=995456d0188ffb51032c28f41527cfa82d87d8a9
ARG COMFYUI_MANAGER_REF=2908ccdd6f567e06040ed42c96f205c09f27b7c8

WORKDIR /opt
RUN git clone https://github.com/Comfy-Org/ComfyUI.git \
    && git -C ComfyUI checkout --quiet ${COMFYUI_REF}

WORKDIR /opt/ComfyUI
RUN python -m pip install -r requirements.txt

RUN git clone https://github.com/Comfy-Org/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager && \
    git -C custom_nodes/ComfyUI-Manager checkout --quiet ${COMFYUI_MANAGER_REF} && \
    pip install -r custom_nodes/ComfyUI-Manager/requirements.txt

# Modelos de vídeo (LTX-2.5) con la misma estructura que su repo de Hugging
# Face: diffusion_models/, text_encoders/, vae/, latent_upscale_models/...
# Se montan en /opt/ComfyUI/models_video y los ve tanto la interfaz como el
# backend de vídeo de oCabra, que reutiliza estos mismos ficheros.
COPY extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml

EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
