#!/usr/bin/env bash
# Post-install script for the modular acestep backend (Bloque 15 Fase 2).
#
# Inputs (set by BackendInstaller):
#   BACKEND_DIR  /data/backends/acestep
#   VENV_DIR     /data/backends/acestep/venv      (created by installer)
#   PYTHON_BIN   /data/backends/acestep/venv/bin/python
#   SRC_DIR      /data/backends/acestep/src       (clone of ace-step/ACE-Step-1.5)
#   BIN_DIR      /data/backends/acestep/bin
#
# What this does:
#   1. Installs `uv` (ACE-Step's build tool) into BIN_DIR if not on PATH.
#   2. Runs `uv sync` against the cloned project, targeting our venv.
#
# torch + torchaudio with CUDA index are seeded by pip_packages BEFORE this
# script runs (so uv sync sees them already installed and doesn't re-resolve).

set -euo pipefail

: "${BACKEND_DIR:?BACKEND_DIR is required}"
: "${VENV_DIR:?VENV_DIR is required}"
: "${PYTHON_BIN:?PYTHON_BIN is required}"
: "${SRC_DIR:?SRC_DIR is required (BackendInstaller must clone ACE-Step first)}"
: "${BIN_DIR:?BIN_DIR is required}"

# 1. uv -- prefer system install, otherwise drop a self-contained binary in BIN_DIR.
UV_BIN="$(command -v uv 2>/dev/null || true)"
if [[ -z "${UV_BIN}" ]]; then
    UV_BIN="${BIN_DIR}/uv"
    if [[ ! -x "${UV_BIN}" ]]; then
        echo "[install_acestep] installing uv into ${BIN_DIR}"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${BIN_DIR}" sh
    fi
fi
echo "[install_acestep] using uv: ${UV_BIN} ($("${UV_BIN}" --version 2>/dev/null || echo unknown))"

# 2. uv sync — point uv at OUR venv so the resolved deps land where the
# backend expects them. ACE-Step pyproject is in SRC_DIR.
cd "${SRC_DIR}"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export UV_PYTHON="${PYTHON_BIN}"

# Try frozen first (uses upstream uv.lock); fall back to fresh resolve if the
# lock is incompatible with our pre-installed torch.
if "${UV_BIN}" sync --frozen; then
    echo "[install_acestep] uv sync --frozen succeeded"
else
    echo "[install_acestep] uv sync --frozen failed; falling back to fresh resolve"
    "${UV_BIN}" sync
fi

# Smoke-check that the package is importable from our venv.
"${PYTHON_BIN}" -c "import acestep; print('acestep version:', getattr(acestep, '__version__', 'unknown'))"

echo "[install_acestep] done."
