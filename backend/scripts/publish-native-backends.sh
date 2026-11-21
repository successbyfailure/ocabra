#!/usr/bin/env bash
# Build the native backends for every GPU/CPU and publish them (see
# backend/Dockerfile.backend-native). Same commits as production by default.
#
#   GHCR_TOKEN=$(gh auth token) GHCR_USER=<user> backend/scripts/publish-native-backends.sh [backend...]
#
# Without arguments: llamacpp_mks llama_cpp whisper_cpp.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GHCR_TOKEN:?set GHCR_TOKEN (e.g. \$(gh auth token))}"

build() {  # backend repo ref script bins cuda
    local backend=$1
    echo "=== ${backend}"
    docker build -f Dockerfile.backend-native \
        --build-arg BACKEND="$1" --build-arg GIT_REPO="$2" --build-arg GIT_REF="$3" \
        --build-arg SCRIPT="$4" --build-arg BINS="$5" --build-arg CUDA="$6" \
        -t "ocabra-backend-build:${backend}" .
    docker run --rm -e GHCR_TOKEN -e GHCR_USER -e BACKEND_OCI_REPOSITORY \
        "ocabra-backend-build:${backend}"
}

for backend in "${@:-llamacpp_mks llama_cpp whisper_cpp}"; do
  for b in $backend; do
    case "$b" in
        llamacpp_mks) build llamacpp_mks https://github.com/makespacemadrid/llama.cpp-mks \
            "${LLAMACPP_MKS_REF:-efbf556487d3f28141f12f4a9aa8a186f02d5753}" \
            install_llama_cpp.sh "server=bin/llama-server" ON ;;
        llama_cpp) build llama_cpp https://github.com/ggml-org/llama.cpp \
            "${LLAMA_CPP_REF:-e6d624b9a4b95c6b52f5ca384d70b75581f9821a}" \
            install_llama_cpp.sh "server=bin/llama-server" ON ;;
        whisper_cpp) build whisper_cpp https://github.com/ggerganov/whisper.cpp \
            "${WHISPER_CPP_REF:-aba02defd6bb6ff5de63721c21e3456645c153e2}" \
            install_whispercpp.sh "server=bin/whisper-server cli=bin/whisper-cli quantize=bin/whisper-quantize" OFF ;;
        *) echo "unknown native backend: $b" >&2; exit 2 ;;
    esac
  done
done
