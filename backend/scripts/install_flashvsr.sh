#!/usr/bin/env bash
# Post-install script for the modular flashvsr backend.
#
# Inputs (set by BackendInstaller):
#   BACKEND_DIR  /data/backends/flashvsr
#   SRC_DIR      /data/backends/flashvsr/src   (clone of OpenImagingLab/FlashVSR)
#   VENV_DIR     /data/backends/flashvsr/venv
#   PYTHON_BIN   ${VENV_DIR}/bin/python
#
# Optional env:
#   FLASHVSR_BUILD_JOBS  default 2. Cada trabajo de nvcc come ~9 GB de RAM.
#
# Instala FlashVSR y compila Block-Sparse-Attention, que es obligatorio.
#
# Dos detalles aprendidos a base de fallos, no cambiar a la ligera:
#
#  1. BLOCK_SPARSE_ATTN_CUDA_ARCHS=80. Su setup.py compila por defecto para
#     80;90;100;110;120, y compute_120 no existe en CUDA 12.4: aborta con
#     "Unsupported gpu architecture". Además no tiene rama para sm_86 (las
#     RTX 30xx de consumo), pero al pedir solo 80 emite también PTX de
#     compatibilidad (arch=compute_80,code=compute_80) que el driver compila
#     en caliente para sm_86. Verificado funcionando en una RTX 3090.
#
#  2. Los pines de torch* de su requirements.txt se ignoran: el venv ya trae
#     la versión que instaló pip_packages y reinstalarla son 3 GB de ruedas
#     idénticas.

set -euo pipefail

: "${BACKEND_DIR:?BACKEND_DIR is required}"
: "${SRC_DIR:?SRC_DIR is required (BackendInstaller must clone FlashVSR first)}"
: "${PYTHON_BIN:?PYTHON_BIN is required}"

FLASHVSR_BUILD_JOBS="${FLASHVSR_BUILD_JOBS:-2}"
# diffsynth fija torch*==2.6.0+cu124, que solo existe en el índice de
# PyTorch: sin este extra-index pip no encuentra torchaudio y aborta.
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
BSA_DIR="${BACKEND_DIR}/block-sparse-attention"

cd "${SRC_DIR}"

echo "[install_flashvsr] instalando FlashVSR y dependencias"
# El setup.py de FlashVSR importa pkg_resources, que ya no viene de serie. Hacen
# falta las dos cosas: setuptools<81 en el venv (que aún trae pkg_resources) y
# --no-build-isolation, porque si no pip monta un entorno de compilación aparte
# con un setuptools moderno donde pkg_resources ya no existe.
"${PYTHON_BIN}" -m pip install "setuptools<81" wheel
grep -vE '^(torch|torchvision|torchaudio)==' requirements.txt > requirements.ocabra.txt
"${PYTHON_BIN}" -m pip install --extra-index-url "${TORCH_INDEX}" -r requirements.ocabra.txt
"${PYTHON_BIN}" -m pip install --extra-index-url "${TORCH_INDEX}" -e . --no-build-isolation
# diffsynth importa modelscope en su downloader y no está en requirements.
"${PYTHON_BIN}" -m pip install modelscope

if [[ ! -d "${BSA_DIR}" ]]; then
    echo "[install_flashvsr] clonando Block-Sparse-Attention"
    git clone --depth 1 https://github.com/mit-han-lab/Block-Sparse-Attention "${BSA_DIR}"
fi

cd "${BSA_DIR}"

# El /usr/local/cuda de la imagen trae nvcc pero no todas las cabeceras: la
# compilación muere con "fatal error: cusparse.h: No such file or directory".
# Las cabeceras que faltan sí vienen en las ruedas de NVIDIA que instala torch,
# así que se añaden al CPATH en vez de meter el toolkit completo por apt.
NVIDIA_DIR="$("${PYTHON_BIN}" -c 'import os, nvidia; print(os.path.dirname(nvidia.__file__))')"
for inc in "${NVIDIA_DIR}"/*/include; do
    [[ -d "${inc}" ]] && CPATH="${inc}${CPATH:+:${CPATH}}"
done
export CPATH
echo "[install_flashvsr] CPATH con cabeceras de NVIDIA: ${CPATH}"

echo "[install_flashvsr] compilando Block-Sparse-Attention (jobs=${FLASHVSR_BUILD_JOBS})"
echo "[install_flashvsr] esto tarda ~45 min: 25 unidades CUDA"
"${PYTHON_BIN}" -m pip install packaging ninja
BLOCK_SPARSE_ATTN_CUDA_ARCHS=80 \
MAX_JOBS="${FLASHVSR_BUILD_JOBS}" \
NVCC_THREADS=4 \
    "${PYTHON_BIN}" setup.py install

echo "[install_flashvsr] comprobando que el módulo carga"
# torch primero: la extensión enlaza contra libc10.so y sin importar torch
# antes el import falla con "libc10.so: cannot open shared object file".
"${PYTHON_BIN}" -c "import torch; import block_sparse_attn; print('block_sparse_attn OK')"

echo "[install_flashvsr] hecho"
