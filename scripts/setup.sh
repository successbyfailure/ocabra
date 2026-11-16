#!/usr/bin/env bash
# Compatibility entry point. The maintained installer lives at repository root.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT_DIR}/setup.sh" "$@"
