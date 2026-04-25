#!/usr/bin/env bash
# Post-install script for the modular llama_cpp backend (Bloque 15 Fase 2).
#
# Inputs (set by BackendInstaller):
#   BACKEND_DIR  /data/backends/llama_cpp
#   SRC_DIR      /data/backends/llama_cpp/src   (clone of ggml-org/llama.cpp)
#   BIN_DIR      /data/backends/llama_cpp/bin
#
# Optional env (read with sensible defaults):
#   LLAMA_CUDA_ARCHITECTURES  "61;70;75;80;86;89"  (RTX 3060 → RTX 4090 + L4/L40S)
#   ENABLE_CUDA               "ON" | "OFF"          (default ON)
#   BUILD_JOBS                int                   (default nproc/2 capped at 8)
#
# Output: ${BIN_DIR}/llama-server, plus any libggml*.so/libllama*.so referenced
# by the binary, copied next to it for LD_LIBRARY_PATH-free loading.
#
# Same recipe as backends/dockerfiles/Dockerfile.llama_cpp, repackaged so the
# slim image can build it at runtime via BackendInstaller.

set -euo pipefail

: "${BACKEND_DIR:?BACKEND_DIR is required}"
: "${SRC_DIR:?SRC_DIR is required (BackendInstaller must clone llama.cpp first)}"
: "${BIN_DIR:?BIN_DIR is required}"

LLAMA_CUDA_ARCHITECTURES="${LLAMA_CUDA_ARCHITECTURES:-61;70;75;80;86;89}"
ENABLE_CUDA="${ENABLE_CUDA:-ON}"
BUILD_JOBS="${BUILD_JOBS:-$(( $(nproc 2>/dev/null || echo 4) / 2 ))}"
[[ "${BUILD_JOBS}" -lt 1 ]] && BUILD_JOBS=1
[[ "${BUILD_JOBS}" -gt 8 ]] && BUILD_JOBS=8

echo "[install_llama_cpp] cmake build (CUDA=${ENABLE_CUDA}, archs=${LLAMA_CUDA_ARCHITECTURES}, jobs=${BUILD_JOBS})"

# CUDA driver stub — the build links against libcuda for symbol resolution but
# the real driver is provided at runtime by the host via /usr/lib/.../libcuda.so.1.
# In the slim base image (no CUDA toolkit) the stub may be missing; if so the
# build silently falls back to CPU. We log the choice so the user knows what
# they got.
if [[ "${ENABLE_CUDA}" == "ON" ]]; then
    if [[ -f /usr/local/cuda/lib64/stubs/libcuda.so ]]; then
        ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
        export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-}"
    elif ! command -v nvcc >/dev/null 2>&1; then
        echo "[install_llama_cpp] WARNING: ENABLE_CUDA=ON but no CUDA toolkit detected; falling back to CPU build."
        ENABLE_CUDA=OFF
    fi
fi

cd "${SRC_DIR}"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA="${ENABLE_CUDA}" \
    -DCMAKE_CUDA_ARCHITECTURES="${LLAMA_CUDA_ARCHITECTURES}" \
    -DLLAMA_CURL=OFF
cmake --build build --target llama-server -j"${BUILD_JOBS}"

mkdir -p "${BIN_DIR}"
cp build/bin/llama-server "${BIN_DIR}/llama-server"
chmod +x "${BIN_DIR}/llama-server"

# Copy libggml/libllama shared libs next to the binary so it loads them
# without LD_LIBRARY_PATH gymnastics at runtime.
find build -type f \( -name "libllama*.so*" -o -name "libggml*.so*" -o -name "libmtmd*.so*" \) \
    -exec cp -a {} "${BIN_DIR}/" \;

echo "[install_llama_cpp] done. Produced:"
ls -la "${BIN_DIR}"
