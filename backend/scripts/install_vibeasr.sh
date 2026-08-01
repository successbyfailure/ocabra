#!/usr/bin/env bash
# Post-install script for the modular vibeasr backend.
#
# Inputs (set by BackendInstaller):
#   BACKEND_DIR  /data/backends/vibeasr
#   SRC_DIR      /data/backends/vibeasr/src   (recursive clone of microsoft/VibeASR.cpp)
#   BIN_DIR      /data/backends/vibeasr/bin
#
# Optional env:
#   VIBEASR_BUILD_JOBS  default nproc capped at 8
#
# Output: ${BIN_DIR}/asr_stream_server + asr_infer + lib*.so deps. CPU-first
# (VibeVoice-ASR-BitNet is a ternary CPU model — no CUDA required).

set -euo pipefail

: "${BACKEND_DIR:?BACKEND_DIR is required}"
: "${SRC_DIR:?SRC_DIR is required (BackendInstaller must clone VibeASR.cpp first)}"
: "${BIN_DIR:?BIN_DIR is required}"

VIBEASR_BUILD_JOBS="${VIBEASR_BUILD_JOBS:-$(( $(nproc 2>/dev/null || echo 4) ))}"
[[ "${VIBEASR_BUILD_JOBS}" -gt 8 ]] && VIBEASR_BUILD_JOBS=8

cd "${SRC_DIR}"

echo "[install_vibeasr] cmake build (jobs=${VIBEASR_BUILD_JOBS})"
cmake -B build \
    -DGGML_AVX2=ON \
    -DGGML_F16C=ON \
    -DGGML_FMA=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --target asr_stream_server --target asr_infer -j"${VIBEASR_BUILD_JOBS}" \
    || cmake --build build -j"${VIBEASR_BUILD_JOBS}"

mkdir -p "${BIN_DIR}"
for bin in asr_stream_server asr_infer; do
    if [[ -f "build/bin/${bin}" ]]; then
        cp "build/bin/${bin}" "${BIN_DIR}/${bin}"
        chmod +x "${BIN_DIR}/${bin}"
    else
        echo "[install_vibeasr] ${bin} not found after build" >&2
        exit 1
    fi
done

# Shared libs live under build/3rdparty/llama.cpp/{src,ggml/src}/, not build/bin.
# Copy them next to the binaries so LD_LIBRARY_PATH=<bin dir> resolves them
# without relying on the build tree's embedded RPATH.
find build -name "libggml*.so*" -o -name "libllama*.so*" -o -name "libllava*.so*" \
    | while read -r lib; do cp -a "${lib}" "${BIN_DIR}/"; done

echo "[install_vibeasr] done. Produced:"
ls -la "${BIN_DIR}"
