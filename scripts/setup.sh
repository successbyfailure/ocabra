#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

set_env_value() {
    local key="$1"
    local value="$2"
    local tmp_file
    tmp_file="$(mktemp)"

    if grep -q "^${key}=" .env; then
        sed "s|^${key}=.*|${key}=${value}|" .env >"$tmp_file"
    else
        cp .env "$tmp_file"
        printf '\n%s=%s\n' "$key" "$value" >>"$tmp_file"
    fi

    mv "$tmp_file" .env
}

random_hex() {
    local length="${1:-32}"

    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex "$length"
        return
    fi

    head -c "$length" /dev/urandom | od -An -tx1 | tr -d ' \n'
}

echo "=== oCabra Setup ==="

# 1. Copy .env if missing
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env from .env.example"
fi

# shellcheck disable=SC1091
set -a
source ./.env
set +a

POSTGRES_USER="${POSTGRES_USER:-ocabra}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
POSTGRES_DB="${POSTGRES_DB:-ocabra}"
MODELS_DIR="${MODELS_DIR:-./data/models}"
HF_CACHE_DIR="${HF_CACHE_DIR:-./data/hf_cache}"
LITELLM_ADMIN_KEY="${LITELLM_ADMIN_KEY:-}"

if [ -z "$POSTGRES_PASSWORD" ] || [ "$POSTGRES_PASSWORD" = "change_me_in_production" ]; then
    POSTGRES_PASSWORD="$(random_hex 24)"
    set_env_value "POSTGRES_PASSWORD" "$POSTGRES_PASSWORD"
    echo "✓ Generated POSTGRES_PASSWORD"
fi

DATABASE_URL="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}"
set_env_value "DATABASE_URL" "$DATABASE_URL"
echo "✓ Synced DATABASE_URL from POSTGRES_*"

if [ -z "$LITELLM_ADMIN_KEY" ]; then
    LITELLM_ADMIN_KEY="$(random_hex 24)"
    set_env_value "LITELLM_ADMIN_KEY" "$LITELLM_ADMIN_KEY"
    echo "✓ Generated LITELLM_ADMIN_KEY"
fi

# 2. Create data directories
mkdir -p "$MODELS_DIR" "$HF_CACHE_DIR"
echo "✓ Data directories ready"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Environment ready"
echo "  Next step: docker compose up -d"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
