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
BSA_DIR="${BACKEND_DIR}/block-sparse-attention"

cd "${SRC_DIR}"

echo "[install_flashvsr] instalando FlashVSR y dependencias"
grep -vE '^(torch|torchvision|torchaudio)==' requirements.txt > requirements.ocabra.txt
"${PYTHON_BIN}" -m pip install -r requirements.ocabra.txt
"${PYTHON_BIN}" -m pip install -e .
# diffsynth importa modelscope en su downloader y no está en requirements.
"${PYTHON_BIN}" -m pip install modelscope

if [[ ! -d "${BSA_DIR}" ]]; then
    echo "[install_flashvsr] clonando Block-Sparse-Attention"
    git clone --depth 1 https://github.com/mit-han-lab/Block-Sparse-Attention "${BSA_DIR}"
fi

cd "${BSA_DIR}"
echo "[install_flashvsr] compilando Block-Sparse-Attention (jobs=${FLASHVSR_BUILD_JOBS})"
echo "[install_flashvsr] esto tarda ~45 min: 25 unidades CUDA"
"${PYTHON_BIN}" -m pip install packaging ninja
BLOCK_SPARSE_ATTN_CUDA_ARCHS=80 \
MAX_JOBS="${FLASHVSR_BUILD_JOBS}" \
NVCC_THREADS=4 \
    "${PYTHON_BIN}" setup.py install

echo "[install_flashvsr] comprobando que el módulo carga"
"${PYTHON_BIN}" -c "import block_sparse_attn; print('block_sparse_attn OK')"

echo "[install_flashvsr] hecho"
