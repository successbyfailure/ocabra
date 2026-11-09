#!/bin/sh
# Instala solamente el Terminal de Strata en un prefijo aislado para el spike.
# El motor y el modelo se descargan despues, al ejecutar `strata launch`.
set -eu

STRATA_RELEASE="v0.2.13"
INSTALLER_SHA256="d47754af1f74c95bf763e0b8d484232443466decd1a22d0be3208760a2f53822"
SPIKE_ROOT="${STRATA_SPIKE_ROOT:-/docker/ai-models/ocabra/models/huggingface/strata-spike}"
FEED_URL="https://github.com/Niko1221/Strata/releases/download/${STRATA_RELEASE}"

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/ocabra-strata-install.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT INT TERM

curl -fsSL --retry 3 "${FEED_URL}/install.sh" -o "$work_dir/install.sh"
actual_sha=$(sha256sum "$work_dir/install.sh" | cut -d' ' -f1)
if [ "$actual_sha" != "$INSTALLER_SHA256" ]; then
  echo "El instalador de Strata no coincide con el SHA-256 fijado." >&2
  echo "Esperado: $INSTALLER_SHA256" >&2
  echo "Recibido: $actual_sha" >&2
  exit 1
fi

mkdir -p "$SPIKE_ROOT/runtime" "$SPIKE_ROOT/bin" "$SPIKE_ROOT/models"
STRATA_FEED="$FEED_URL" \
STRATA_DATA_DIR="$SPIKE_ROOT/runtime" \
STRATA_BIN_DIR="$SPIKE_ROOT/bin" \
  sh "$work_dir/install.sh"

"$SPIKE_ROOT/bin/strata" --version
echo "Terminal listo: $SPIKE_ROOT/bin/strata"
echo "Los modelos se guardaran en: $SPIKE_ROOT/models"
