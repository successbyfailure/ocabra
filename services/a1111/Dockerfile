FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GIT_TERMINAL_PROMPT=0 \
    COMMANDLINE_ARGS="--listen --api --port 7860 --skip-torch-cuda-test"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -s /bin/bash a1111

WORKDIR /opt
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git \
    && chown -R 1000:1000 /opt/stable-diffusion-webui

WORKDIR /opt/stable-diffusion-webui

# Pre-clone all sub-repositories that A1111 needs at runtime.
# Stability-AI/stablediffusion was deleted; use a mirror that has the same commit.
# Checking out the exact commit hash that launch_utils.py expects means A1111 will
# see dir+hash already correct and skip the git_clone step entirely at startup.
RUN git clone https://github.com/Jonel865/stable-diffusion-stability-ai.git \
        repositories/stable-diffusion-stability-ai && \
    git -C repositories/stable-diffusion-stability-ai checkout cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf && \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git \
        repositories/stable-diffusion-webui-assets && \
    git -C repositories/stable-diffusion-webui-assets checkout 6f7db241d2f8ba7457bac5ca9753331f0c266917 && \
    git clone https://github.com/Stability-AI/generative-models.git \
        repositories/generative-models && \
    git -C repositories/generative-models checkout 45c443b316737a4ab6e40413d7794a7f5657c19f && \
    git clone https://github.com/crowsonkb/k-diffusion.git \
        repositories/k-diffusion && \
    git -C repositories/k-diffusion checkout ab527a9a6d347f364e3d185ba6d714e22d80cb3c && \
    git clone https://github.com/salesforce/BLIP.git \
        repositories/BLIP && \
    git -C repositories/BLIP checkout 48211a1594f1321b00f14c9f7a5b4813144b2fb9 && \
    chown -R 1000:1000 /opt/stable-diffusion-webui/repositories

RUN python -m pip install --upgrade pip setuptools wheel

# setuptools 80+ removed pkg_resources from the package; CLIP's setup.py requires it.
# Write a constraints file that caps setuptools<80, then set PIP_CONSTRAINT so every
# pip invocation — including the build-isolation env pip creates for CLIP — respects it.
RUN echo "setuptools<80" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

USER 1000
RUN git config --global credential.helper "" && \
    python -m venv venv && \
    venv/bin/pip install --upgrade pip "setuptools<80" wheel

EXPOSE 7860

CMD ["bash", "-lc", "./webui.sh"]
