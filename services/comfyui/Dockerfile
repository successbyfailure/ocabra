FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pinned (INF-07) to upstream HEAD as of 2026-11-19 (no image of this service
# runs on the hub to copy the commit from). Override with --build-arg.
ARG COMFYUI_REF=995456d0188ffb51032c28f41527cfa82d87d8a9
ARG COMFYUI_MANAGER_REF=2908ccdd6f567e06040ed42c96f205c09f27b7c8

WORKDIR /opt
RUN git clone https://github.com/Comfy-Org/ComfyUI.git \
    && git -C ComfyUI checkout --quiet ${COMFYUI_REF}

WORKDIR /opt/ComfyUI
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

RUN git clone https://github.com/Comfy-Org/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager && \
    git -C custom_nodes/ComfyUI-Manager checkout --quiet ${COMFYUI_MANAGER_REF} && \
    pip install -r custom_nodes/ComfyUI-Manager/requirements.txt

EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
