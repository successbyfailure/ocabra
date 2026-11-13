#!/usr/bin/env bash
# Post-install script for the modular llama_cpp backend (Bloque 15 Fase 2).
#
# Inputs (set by BackendInstaller):
#   BACKEND_DIR  /data/backends/llama_cpp
#   SRC_DIR      /data/backends/llama_cpp/src   (clone of ggml-org/llama.cpp)
#   BIN_DIR      /data/backends/llama_cpp/bin
#
# Optional env (read with sensible defaults):
#   LLAMA_CUDA_ARCHITECTURES  "75;80;86;89"  (RTX 2060/Turing → RTX 4090 + L4/L40S)
#   ENABLE_CUDA               "ON" | "OFF"   (default ON)
#   BUILD_JOBS                int            (default nproc/2 capped at 8)
#   LLAMA_CUDA_NCCL           "ON" | "OFF"   (default ON with CUDA): NCCL for the
#                             cross-GPU AllReduce of --split-mode tensor
#   NCCL_VERSION              pip version of nvidia-nccl-cu<major> (default pinned)
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

# 61/70 (Pascal/Volta) dropped from the default: CUDA 13's nvcc
# (`nvcc --list-gpu-arch`) no longer accepts them at all — "nvcc fatal:
# Unsupported gpu architecture 'compute_61'" fails the whole build even
# though real targets like 86 are listed alongside it. 75 (Turing) is
# nvcc 13's actual floor.
LLAMA_CUDA_ARCHITECTURES="${LLAMA_CUDA_ARCHITECTURES:-75;80;86;89}"
ENABLE_CUDA="${ENABLE_CUDA:-ON}"
BUILD_JOBS="${BUILD_JOBS:-$(( $(nproc 2>/dev/null || echo 4) / 2 ))}"
[[ "${BUILD_JOBS}" -lt 1 ]] && BUILD_JOBS=1
[[ "${BUILD_JOBS}" -gt 8 ]] && BUILD_JOBS=8

LLAMA_CUDA_NCCL="${LLAMA_CUDA_NCCL:-ON}"
NCCL_VERSION="${NCCL_VERSION:-2.31.2}"

echo "[install_llama_cpp] cmake build (CUDA=${ENABLE_CUDA}, archs=${LLAMA_CUDA_ARCHITECTURES}, jobs=${BUILD_JOBS}, nccl=${LLAMA_CUDA_NCCL})"

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

# NCCL. Without it, --split-mode tensor sums each layer's partial results
# across GPUs through pinned host memory, by design, even when the GPUs can
# reach each other peer-to-peer. With it and P2P (3060 + 3090, 2026-11-13):
# prefill +9% at 35k tokens and +6% at 122k, decode unchanged, +160-200 MB of
# VRAM per GPU from load. Layer split and single-GPU models do not use it.
# NCCL is not in the image: NVIDIA's wheel for this CUDA major provides the
# header and the library, kept in ${BACKEND_DIR}/nccl (the build links it by
# absolute path). If it cannot be fetched the build goes on without it.
# Explicit OFF otherwise: CMake's cache would keep an earlier ON.
NCCL_FLAGS=(-DGGML_CUDA_NCCL=OFF)
if [[ "${ENABLE_CUDA}" == "ON" && "${LLAMA_CUDA_NCCL}" == "ON" ]]; then
    CUDA_MAJOR="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1)"
    NCCL_DIR="${BACKEND_DIR}/nccl"
    if [[ ! -f "${NCCL_DIR}/include/nccl.h" || ! -f "${NCCL_DIR}/lib/libnccl.so.2" ]] \
        && [[ -n "${CUDA_MAJOR}" ]]; then
        WHEEL_DIR="$(mktemp -d)"
        if python3 -m pip download --no-deps --only-binary=:all: --quiet \
                "nvidia-nccl-cu${CUDA_MAJOR}==${NCCL_VERSION}" -d "${WHEEL_DIR}" \
            && python3 -I -m zipfile -e "${WHEEL_DIR}"/nvidia_nccl_*.whl "${WHEEL_DIR}/x"; then
            rm -rf "${NCCL_DIR}.new"
            mkdir -p "${NCCL_DIR}.new"
            cp -r "${WHEEL_DIR}/x/nvidia/nccl/include" "${WHEEL_DIR}/x/nvidia/nccl/lib" "${NCCL_DIR}.new/"
            ln -sf libnccl.so.2 "${NCCL_DIR}.new/lib/libnccl.so"
            rm -rf "${NCCL_DIR}" && mv "${NCCL_DIR}.new" "${NCCL_DIR}"
        else
            echo "[install_llama_cpp] WARNING: could not fetch nvidia-nccl-cu${CUDA_MAJOR}==${NCCL_VERSION}; building without NCCL."
        fi
        rm -rf "${WHEEL_DIR}"
    fi
    if [[ -f "${NCCL_DIR}/include/nccl.h" ]]; then
        NCCL_FLAGS=(-DGGML_CUDA_NCCL=ON "-DNCCL_ROOT=${NCCL_DIR}")
    fi
fi

cd "${SRC_DIR}"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA="${ENABLE_CUDA}" \
    -DCMAKE_CUDA_ARCHITECTURES="${LLAMA_CUDA_ARCHITECTURES}" \
    -DLLAMA_CURL=OFF \
    "${NCCL_FLAGS[@]}"
cmake --build build --target llama-server -j"${BUILD_JOBS}"

mkdir -p "${BIN_DIR}"

# Install each file by copying it beside its destination and renaming it over.
# Reinstalling runs while workers are serving: copying straight over a running
# executable fails with "Text file busy", and overwriting a loaded .so in place
# corrupts the processes that mapped it. A rename swaps the directory entry and
# leaves running processes on the old inode.
install_atomic() {
    local src="$1" dest="$2"
    cp -a "${src}" "${dest}.new.$$"
    mv -f "${dest}.new.$$" "${dest}"
}

install_atomic build/bin/llama-server "${BIN_DIR}/llama-server"
chmod +x "${BIN_DIR}/llama-server"

# Copy libggml/libllama shared libs next to the binary so it loads them
# without LD_LIBRARY_PATH gymnastics at runtime.
while IFS= read -r -d '' lib; do
    install_atomic "${lib}" "${BIN_DIR}/$(basename "${lib}")"
done < <(find build -type f \( -name "libllama*.so*" -o -name "libggml*.so*" -o -name "libmtmd*.so*" \) -print0)

echo "[install_llama_cpp] done. Produced:"
ls -la "${BIN_DIR}"
