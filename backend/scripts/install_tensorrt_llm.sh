#!/usr/bin/env bash
# TensorRT-LLM for the tensorrt_llm backend in its "docker" launch mode: the
# runtime is NVIDIA's NGC image (TENSORRT_LLM_DOCKER_IMAGE), started as a
# sibling container per model. Installing = pulling that image (tens of GB, in
# Docker's storage, not in BACKEND_DIR) and recording which one it was.
# Needs the docker CLI and socket (the compose stack has both; the ocabra-node
# image does not, so it cannot install this backend).
#
# Uninstalling the backend does not remove the image: `docker rmi` it by hand.
set -euo pipefail

: "${BACKEND_DIR:?BACKEND_DIR not set}"
image="${TENSORRT_LLM_DOCKER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:latest}"
docker_bin="${TENSORRT_LLM_DOCKER_BIN:-docker}"
command -v "$docker_bin" >/dev/null 2>&1 || [ -x "$docker_bin" ] \
    || { echo "[install_tensorrt_llm] no docker CLI here: TensorRT-LLM runs as an NGC container" >&2; exit 1; }
[ -S /var/run/docker.sock ] \
    || { echo "[install_tensorrt_llm] no /var/run/docker.sock: TensorRT-LLM runs as an NGC container" >&2; exit 1; }

echo "[install_tensorrt_llm] pulling ${image} (large: tens of GB)"
"$docker_bin" pull "$image"
digest="$("$docker_bin" image inspect --format '{{index .RepoDigests 0}}' "$image" 2>/dev/null || echo "$image")"
mkdir -p "${BACKEND_DIR}/bin"
printf '%s\n' "$digest" > "${BACKEND_DIR}/IMAGE"
# extra_bins needs a file: a launcher that runs trtllm-serve from the image
# (handy for checks; the backend itself starts the container its own way).
cat > "${BACKEND_DIR}/bin/trtllm-serve" <<SH
#!/bin/sh
exec "$docker_bin" run --rm --gpus all "$image" trtllm-serve "\$@"
SH
chmod +x "${BACKEND_DIR}/bin/trtllm-serve"
echo "[install_tensorrt_llm] ${digest}"
