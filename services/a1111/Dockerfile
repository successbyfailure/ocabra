FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COMMANDLINE_ARGS="--listen --api --port 7860 --skip-torch-cuda-test"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

WORKDIR /opt/stable-diffusion-webui
RUN python -m pip install --upgrade pip setuptools wheel

EXPOSE 7860

CMD ["bash", "-lc", "./webui.sh"]
