#!/usr/bin/env bash
set -euo pipefail

BITNET_COMMIT="${BITNET_COMMIT:-main}"
WORKDIR="${WORKDIR:-/build/BitNet}"
OUTDIR="${OUTDIR:-/out}"
BITNET_PRETUNED_MODEL="${BITNET_PRETUNED_MODEL:-bitnet_b1_58-3B}"
BITNET_KERNEL_FLAVOR="${BITNET_KERNEL_FLAVOR:-tl2}"
BITNET_BUILD_JOBS="${BITNET_BUILD_JOBS:-8}"
BITNET_ENABLE_CUDA="${BITNET_ENABLE_CUDA:-false}"
BITNET_BUILD_PRISMML="${BITNET_BUILD_PRISMML:-true}"
PRISMML_WORKDIR="${PRISMML_WORKDIR:-/build/prismml-llama}"

apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  python3 \
  python3-pip
rm -rf /var/lib/apt/lists/*

python3 -m pip install --no-cache-dir huggingface-hub numpy

git clone --recursive --depth 1 --branch "${BITNET_COMMIT}" https://github.com/microsoft/BitNet.git "${WORKDIR}"
cd "${WORKDIR}"

KERNEL_HEADER="preset_kernels/${BITNET_PRETUNED_MODEL}/bitnet-lut-kernels-${BITNET_KERNEL_FLAVOR}.h"
if [[ ! -f "${KERNEL_HEADER}" ]]; then
  echo "Missing pretuned kernel header: ${KERNEL_HEADER}" >&2
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

cmake -B build \
  -DGGML_CUDA="${GGML_CUDA_FLAG}" \
  -DGGML_AVX2=ON \
  -DGGML_F16C=ON \
  -DGGML_FMA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j"${BITNET_BUILD_JOBS}"

mkdir -p "${OUTDIR}"
if [[ -f build/bin/llama-server ]]; then
  cp build/bin/llama-server "${OUTDIR}/bitnet-server"
elif [[ -f build/bin/server ]]; then
  cp build/bin/server "${OUTDIR}/bitnet-server"
else
  echo "llama-server binary not found after build" >&2
  exit 1
fi
chmod +x "${OUTDIR}/bitnet-server"

LIB_OUTDIR="${OUTDIR}/lib"
mkdir -p "${LIB_OUTDIR}"

# Copy non-system shared libs produced by BitNet/llama.cpp so runtime can load bitnet-server.
find build -type f \( -name "libllama.so*" -o -name "libggml*.so*" \) -exec cp -a {} "${LIB_OUTDIR}/" \;

if [[ -z "$(ls -A "${LIB_OUTDIR}" 2>/dev/null)" ]]; then
  echo "No BitNet shared libraries were exported to ${LIB_OUTDIR}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# PrismML llama.cpp fork for Bonsai 27B (Q1_0_g128 hybrid-attention kernels).
# Exported to ${OUTDIR}/prismml/bonsai-server with its libs alongside; the
# Dockerfile reshapes it into /backend/bin/prismml/. Disable with
# BITNET_BUILD_PRISMML=false.
# ---------------------------------------------------------------------------
if [[ "${BITNET_BUILD_PRISMML}" == "true" ]]; then
  git clone --depth 1 https://github.com/PrismML-Eng/llama.cpp.git "${PRISMML_WORKDIR}"
  cd "${PRISMML_WORKDIR}"
  cmake -B build \
    -DGGML_CUDA="${GGML_CUDA_FLAG}" \
    -DGGML_AVX2=ON \
    -DGGML_F16C=ON \
    -DGGML_FMA=ON \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target llama-server -j"${BITNET_BUILD_JOBS}"

  PRISMML_OUTDIR="${OUTDIR}/prismml"
  mkdir -p "${PRISMML_OUTDIR}"
  if [[ -f build/bin/llama-server ]]; then
    cp build/bin/llama-server "${PRISMML_OUTDIR}/bonsai-server"
  elif [[ -f build/bin/server ]]; then
    cp build/bin/server "${PRISMML_OUTDIR}/bonsai-server"
  else
    echo "PrismML llama-server binary not found after build" >&2
    exit 1
  fi
  chmod +x "${PRISMML_OUTDIR}/bonsai-server"
  # Preserve soname symlinks (libggml.so.0 -> .0.x): copy build/bin/*.so* rather
  # than `find -type f`, which drops the symlinks and breaks the loader.
  cp -a build/bin/*.so* "${PRISMML_OUTDIR}/" 2>/dev/null || true
  # For CUDA builds, bundle cuBLAS/cuBLASLt/cudart/NCCL next to the binary: the
  # target container provides only libcuda.so.1 (driver) via the NVIDIA runtime.
  if [[ "${GGML_CUDA_FLAG}" == "ON" ]]; then
    for pat in "libcublas.so.*" "libcublasLt.so.*" "libcudart.so.*"; do
      find /usr/local/cuda/lib64 -maxdepth 1 -name "${pat}" -exec cp -a {} "${PRISMML_OUTDIR}/" \; 2>/dev/null || true
    done
    nccl="$(find / -name 'libnccl.so.2*' 2>/dev/null | head -1)"
    [[ -n "${nccl}" ]] && cp -a "${nccl}" "${PRISMML_OUTDIR}/" || true
  fi
fi
