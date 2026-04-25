#!/usr/bin/env bash
# Post-install script for the modular bitnet backend (Bloque 15 Fase 2).
#
# Inputs (set by BackendInstaller):
#   BACKEND_DIR  /data/backends/bitnet
#   SRC_DIR      /data/backends/bitnet/src   (recursive clone of microsoft/BitNet)
#   BIN_DIR      /data/backends/bitnet/bin
#
# Optional env:
#   BITNET_PRETUNED_MODEL  default bitnet_b1_58-3B
#   BITNET_KERNEL_FLAVOR   default tl2
#   BITNET_BUILD_JOBS      default nproc/2 capped at 8
#   BITNET_ENABLE_CUDA     default false   (BitNet's pretuned kernels are CPU-first)
#
# Output: ${BIN_DIR}/bitnet-server (renamed from llama-server) + lib*.so deps.
#
# Adapted from backend/scripts/build_bitnet.sh: that script handled clone+build
# in one go for the fat Dockerfile; here BackendInstaller already clones into
# SRC_DIR, so we only do the kernel-header copy + cmake build.

set -euo pipefail

: "${BACKEND_DIR:?BACKEND_DIR is required}"
: "${SRC_DIR:?SRC_DIR is required (BackendInstaller must clone BitNet first)}"
: "${BIN_DIR:?BIN_DIR is required}"

BITNET_PRETUNED_MODEL="${BITNET_PRETUNED_MODEL:-bitnet_b1_58-3B}"
BITNET_KERNEL_FLAVOR="${BITNET_KERNEL_FLAVOR:-tl2}"
BITNET_BUILD_JOBS="${BITNET_BUILD_JOBS:-$(( $(nproc 2>/dev/null || echo 4) / 2 ))}"
[[ "${BITNET_BUILD_JOBS}" -lt 1 ]] && BITNET_BUILD_JOBS=1
[[ "${BITNET_BUILD_JOBS}" -gt 8 ]] && BITNET_BUILD_JOBS=8
BITNET_ENABLE_CUDA="${BITNET_ENABLE_CUDA:-false}"

cd "${SRC_DIR}"

# BitNet ships pretuned LUT kernel headers per (model, flavor) combo. Copy
# the right ones into include/ before cmake, otherwise the build fails.
KERNEL_HEADER="preset_kernels/${BITNET_PRETUNED_MODEL}/bitnet-lut-kernels-${BITNET_KERNEL_FLAVOR}.h"
if [[ ! -f "${KERNEL_HEADER}" ]]; then
    echo "[install_bitnet] missing pretuned kernel header: ${KERNEL_HEADER}" >&2
    exit 1
fi
cp "${KERNEL_HEADER}" include/bitnet-lut-kernels.h

KERNEL_CONFIG="preset_kernels/${BITNET_PRETUNED_MODEL}/kernel_config_${BITNET_KERNEL_FLAVOR}.ini"
if [[ -f "${KERNEL_CONFIG}" ]]; then
    cp "${KERNEL_CONFIG}" include/kernel_config.ini
fi

if [[ "${BITNET_ENABLE_CUDA}" == "true" && -f /usr/local/cuda/lib64/stubs/libcuda.so ]]; then
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
    export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-}"
    GGML_CUDA_FLAG="ON"
else
    GGML_CUDA_FLAG="OFF"
fi

echo "[install_bitnet] cmake build (CUDA=${GGML_CUDA_FLAG}, model=${BITNET_PRETUNED_MODEL}, kernel=${BITNET_KERNEL_FLAVOR}, jobs=${BITNET_BUILD_JOBS})"

cmake -B build \
    -DGGML_CUDA="${GGML_CUDA_FLAG}" \
    -DGGML_AVX2=ON \
    -DGGML_F16C=ON \
    -DGGML_FMA=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j"${BITNET_BUILD_JOBS}"

mkdir -p "${BIN_DIR}"
if [[ -f build/bin/llama-server ]]; then
    cp build/bin/llama-server "${BIN_DIR}/bitnet-server"
elif [[ -f build/bin/server ]]; then
    cp build/bin/server "${BIN_DIR}/bitnet-server"
else
    echo "[install_bitnet] llama-server binary not found after build" >&2
    exit 1
fi
chmod +x "${BIN_DIR}/bitnet-server"

# Copy non-system shared libs produced by BitNet/llama.cpp so runtime can load
# bitnet-server without LD_LIBRARY_PATH gymnastics.
find build -type f \( -name "libllama.so*" -o -name "libggml*.so*" \) \
    -exec cp -a {} "${BIN_DIR}/" \;

echo "[install_bitnet] done. Produced:"
ls -la "${BIN_DIR}"
