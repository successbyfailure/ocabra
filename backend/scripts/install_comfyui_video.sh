#!/usr/bin/env bash
# Post-install script for the modular comfyui_video backend.
#
# Inputs (set by BackendInstaller):
#   BACKEND_DIR  /data/backends/comfyui_video
#   SRC_DIR      /data/backends/comfyui_video/src   (clone of Comfy-Org/ComfyUI)
#   VENV_DIR     /data/backends/comfyui_video/venv
#   PYTHON_BIN   ${VENV_DIR}/bin/python
#
# Instala las dependencias de ComfyUI sin tocar el torch que ya trae el venv
# (pip_packages lo instaló desde el índice cu130; reinstalarlo desde PyPI
# bajaría la variante equivocada y 3 GB de ruedas).

set -euo pipefail

: "${SRC_DIR:?SRC_DIR is required (BackendInstaller must clone ComfyUI first)}"
: "${PYTHON_BIN:?PYTHON_BIN is required}"

cd "${SRC_DIR}"
grep -viE '^(torch|torchvision|torchaudio)([<>=~ ].*)?$' requirements.txt > /tmp/comfyui_video_requirements.txt
"${PYTHON_BIN}" -m pip install --no-cache-dir -r /tmp/comfyui_video_requirements.txt
rm -f /tmp/comfyui_video_requirements.txt

# El worker de oCabra habla con ComfyUI por HTTP y websocket.
"${PYTHON_BIN}" -m pip install --no-cache-dir httpx aiohttp

"${PYTHON_BIN}" -c "import torch, comfy; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
