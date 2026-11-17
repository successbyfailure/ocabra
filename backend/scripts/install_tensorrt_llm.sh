#!/usr/bin/env bash
# TensorRT-LLM for the tensorrt_llm backend in its "docker" launch mode: the
# runtime is NVIDIA's NGC image (TENSORRT_LLM_DOCKER_IMAGE), started as a
# sibling container per model. Installing = pulling that image (tens of GB, in
# Docker's storage, not in BACKEND_DIR) and recording which one it was.
#
# The api has no Docker socket (SEG-03): the pull is done by the docker agent
# (operation trtllm.pull_image, with the image of its own settings). The
# ocabra-node image has no agent, so it cannot install this backend.
#
# Uninstalling the backend does not remove the image: `docker rmi` it by hand.
set -euo pipefail

: "${BACKEND_DIR:?BACKEND_DIR not set}"

echo "[install_tensorrt_llm] pulling the TensorRT-LLM image through the docker agent (large: tens of GB)"
digest="$(python - <<'PY'
import asyncio
import sys

from ocabra.core import docker_ops

if not docker_ops.available():
    sys.exit("[install_tensorrt_llm] no docker agent here: TensorRT-LLM runs as an NGC container")
result = asyncio.run(docker_ops.call("trtllm.pull_image"))
if result.get("code"):
    sys.exit(f"[install_tensorrt_llm] pull failed: {result.get('stderr') or result.get('stdout')}")
print(result["digest"])
PY
)"
mkdir -p "${BACKEND_DIR}/bin"
printf '%s\n' "$digest" > "${BACKEND_DIR}/IMAGE"
# extra_bins needs a file. The engines are served by the docker agent
# (operation trtllm.serve), not from here.
cat > "${BACKEND_DIR}/bin/trtllm-serve" <<'SH'
#!/bin/sh
echo "trtllm-serve runs in the TensorRT-LLM image through oCabra's docker agent." >&2
exit 1
SH
chmod +x "${BACKEND_DIR}/bin/trtllm-serve"
echo "[install_tensorrt_llm] ${digest}"
