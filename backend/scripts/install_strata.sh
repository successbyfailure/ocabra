#!/usr/bin/env bash
# Strata Terminal ("control" bundle: its own Python 3.12 + serve/server.py) for
# the strata backend. Downloads the pinned release asset and checks its
# SHA-256 instead of running upstream's install.sh (no curl | sh).
#
# The engine and the model weights are NOT installed here: they depend on the
# model and the GPU, and `strata launch` fetches them when a model is prepared
# (its run config then points at them; see docs/tasks/strata-backend-spike-plan.md).
#
# Env from BackendInstaller: BACKEND_DIR, BIN_DIR.
set -euo pipefail

STRATA_RELEASE="${STRATA_RELEASE:-v0.2.13}"
STRATA_CONTROL_VERSION="${STRATA_CONTROL_VERSION:-0.1.59.1}"
STRATA_CONTROL_SHA256="${STRATA_CONTROL_SHA256:-73fb612848cf00fa1072623029b35405bbbb63e3f4c688fe6f38c78e3fc022ac}"
asset="strata-control-${STRATA_CONTROL_VERSION}-linux-x86_64.tar.gz"
url="https://github.com/Niko1221/Strata/releases/download/${STRATA_RELEASE}/${asset}"

: "${BACKEND_DIR:?BACKEND_DIR not set}"
dest="${BACKEND_DIR}/terminal"
work="$(mktemp -d "${BACKEND_DIR}/.strata-install.XXXXXX")"
trap 'rm -rf "$work"' EXIT

echo "[install_strata] downloading ${asset} (${STRATA_RELEASE})"
curl -fsSL --retry 3 "$url" -o "$work/$asset"
actual="$(sha256sum "$work/$asset" | cut -d' ' -f1)"
if [ "$actual" != "$STRATA_CONTROL_SHA256" ]; then
    echo "[install_strata] SHA-256 mismatch: expected ${STRATA_CONTROL_SHA256}, got ${actual}" >&2
    exit 1
fi
mkdir -p "$work/x"
tar -xzf "$work/$asset" -C "$work/x"
# One top-level folder in the archive; tolerate it being flat.
src="$work/x"
if [ "$(find "$src" -mindepth 1 -maxdepth 1 | wc -l)" = 1 ] && [ -d "$(find "$src" -mindepth 1 -maxdepth 1)" ]; then
    src="$(find "$src" -mindepth 1 -maxdepth 1)"
fi
for need in python/bin/python3.12 serve/server.py; do
    [ -e "$src/$need" ] || { echo "[install_strata] archive has no $need" >&2; exit 1; }
done
rm -rf "$dest.new" && mv "$src" "$dest.new"
rm -rf "$dest" && mv "$dest.new" "$dest"
"$dest/python/bin/python3.12" -c 'import sys; print("[install_strata] python", sys.version.split()[0])'
echo "[install_strata] Strata Terminal ${STRATA_CONTROL_VERSION} in ${dest}"
