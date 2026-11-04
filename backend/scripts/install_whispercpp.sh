#!/usr/bin/env bash
# Post-install script for the modular whisper_cpp backend.
#
# Inputs (set by BackendInstaller):
#   BACKEND_DIR  /data/backends/whisper_cpp
#   SRC_DIR      /data/backends/whisper_cpp/src   (clone of ggerganov/whisper.cpp)
#   BIN_DIR      /data/backends/whisper_cpp/bin
#
# Optional env:
#   BUILD_JOBS   int  (default nproc/2 capped at 8)
#
# CPU-only on purpose: this backend exists to absorb STT load when the GPU
# whisper variants (recomendado-stt's higher-priority targets) are busy, not
# to compete with them for VRAM. No CUDA build flag — never ON.
#
# Output: ${BIN_DIR}/{whisper-server,whisper-cli,whisper-quantize}, plus any
# libggml*/libwhisper*.so referenced by them, copied next to the binaries for
# LD_LIBRARY_PATH-free loading.

set -euo pipefail

: "${BACKEND_DIR:?BACKEND_DIR is required}"
: "${SRC_DIR:?SRC_DIR is required (BackendInstaller must clone whisper.cpp first)}"
: "${BIN_DIR:?BIN_DIR is required}"

BUILD_JOBS="${BUILD_JOBS:-$(( $(nproc 2>/dev/null || echo 4) / 2 ))}"
[[ "${BUILD_JOBS}" -lt 1 ]] && BUILD_JOBS=1
[[ "${BUILD_JOBS}" -gt 8 ]] && BUILD_JOBS=8

echo "[install_whispercpp] cmake build (CPU-only, jobs=${BUILD_JOBS})"

cd "${SRC_DIR}"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DWHISPER_BUILD_SERVER=ON
cmake --build build --target whisper-server whisper-cli whisper-quantize -j"${BUILD_JOBS}"

mkdir -p "${BIN_DIR}"
for bin in whisper-server whisper-cli whisper-quantize; do
    cp "build/bin/${bin}" "${BIN_DIR}/${bin}"
    chmod +x "${BIN_DIR}/${bin}"
done

# Copy libggml/libwhisper shared libs next to the binaries so they load
# without LD_LIBRARY_PATH gymnastics at runtime.
find build -type f \( -name "libggml*.so*" -o -name "libwhisper*.so*" \) \
    -exec cp -a {} "${BIN_DIR}/" \;

echo "[install_whispercpp] done. Produced:"
ls -la "${BIN_DIR}"
