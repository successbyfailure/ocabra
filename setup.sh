#!/usr/bin/env bash
# oCabra — first-run installation for Linux and Windows/WSL2.
# Usage: bash setup.sh [--with-services]
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }
fatal() { err "$@"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WITH_SERVICES=false
for arg in "$@"; do
    case "$arg" in
        --with-services) WITH_SERVICES=true ;;
        -h|--help)
            echo "Usage: bash setup.sh [--with-services]"
            echo "  --with-services  Build and start optional UI/model services whose sources exist."
            exit 0
            ;;
        *) fatal "Unknown option: $arg" ;;
    esac
done

IS_WSL=false
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    IS_WSL=true
fi

env_value() {
    local key="$1"
    awk -v key="$key" 'index($0, key "=") == 1 {sub(/^[^=]*=/, ""); value=$0} END {print value}' .env
}

set_env_value() {
    local key="$1" value="$2" tmp_file
    tmp_file="$(mktemp "./.env.XXXXXX")"
    awk -v key="$key" -v value="$value" '
        BEGIN {found=0}
        index($0, key "=") == 1 {if (!found) print key "=" value; found=1; next}
        {print}
        END {if (!found) print key "=" value}
    ' .env >"$tmp_file"
    chmod --reference=.env "$tmp_file" 2>/dev/null || true
    mv "$tmp_file" .env
}

random_hex() {
    local bytes="${1:-32}"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "import secrets; print(secrets.token_hex(${bytes}))"
    elif command -v openssl >/dev/null 2>&1; then
        openssl rand -hex "$bytes"
    else
        head -c "$bytes" /dev/urandom | od -An -tx1 | tr -d ' \n'
    fi
}

urlencode() {
    local value="$1"
    if command -v python3 >/dev/null 2>&1; then
        VALUE_TO_ENCODE="$value" python3 -c \
            'import os, urllib.parse; print(urllib.parse.quote(os.environ["VALUE_TO_ENCODE"], safe=""))'
    else
        printf '%s' "$value"
    fi
}

echo -e "\n${BOLD}=== oCabra Setup ===${NC}\n"
if $IS_WSL; then
    info "WSL2 detected; enabling Windows-compatible installation defaults."
    # A Windows drive (/mnt/c, /mnt/j…) can't hold Linux permissions: Postgres
    # refuses to start («could not change permissions of directory»), the api
    # (uid 1000) can't own its data, and Docker is very slow there.
    case "$(pwd -P)" in
        /mnt/[a-zA-Z]|/mnt/[a-zA-Z]/*)
            fatal "oCabra is on a Windows drive ($(pwd -P)). Clone it inside WSL instead, e.g.:
         mkdir -p ~/src && cd ~/src && git clone https://github.com/successbyfailure/ocabra && cd ocabra && bash setup.sh"
            ;;
    esac
fi

command -v docker >/dev/null 2>&1 || fatal "Docker is not installed. See https://docs.docker.com/engine/install/"
docker info >/dev/null 2>&1 || fatal "The Docker daemon is not reachable. Start Docker Desktop or Docker Engine."
ok "Docker found: $(docker --version)"

docker compose version >/dev/null 2>&1 \
    || fatal "Docker Compose v2.30+ is required. See https://docs.docker.com/compose/install/"
COMPOSE=(docker compose)
COMPOSE_VERSION="$(docker compose version --short | sed 's/^v//')"
if [ "$(printf '%s\n' 2.30.0 "$COMPOSE_VERSION" | sort -V | head -1)" != "2.30.0" ]; then
    fatal "Docker Compose 2.30+ is required for portable GPU requests (found ${COMPOSE_VERSION})."
fi
ok "Docker Compose found: ${COMPOSE_VERSION}"
command -v curl >/dev/null 2>&1 || fatal "curl is required by the setup health check."

# WSL exposes the Windows driver here even when nvidia-smi is not on PATH.
NVIDIA_SMI="$(command -v nvidia-smi || true)"
if [ -z "$NVIDIA_SMI" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    NVIDIA_SMI=/usr/lib/wsl/lib/nvidia-smi
fi
[ -n "$NVIDIA_SMI" ] || fatal "nvidia-smi not found. Install or update the NVIDIA host driver first."
ok "NVIDIA driver: $($NVIDIA_SMI --query-gpu=driver_version --format=csv,noheader | head -1)"

# Test the capability instead of looking for Linux-specific toolkit files.
GPU_TEST_IMAGE="${OCABRA_GPU_TEST_IMAGE:-nvidia/cuda:12.4.1-base-ubuntu22.04}"
info "Checking CUDA access from a Linux container..."
if ! docker run --rm --gpus all "$GPU_TEST_IMAGE" nvidia-smi >/dev/null 2>&1; then
    if $IS_WSL; then
        fatal "Docker cannot access the GPU. Update WSL and the Windows NVIDIA driver, then enable Docker Desktop's WSL2 backend."
    fi
    fatal "Docker cannot access the GPU. Check NVIDIA Container Toolkit configuration."
fi
ok "CUDA is available inside Docker"

info "Detected GPUs:"
$NVIDIA_SMI --query-gpu=index,name,memory.total --format=csv,noheader | while IFS= read -r line; do
    echo "      GPU $line"
done

# Create configuration before resolving host paths so .env is authoritative.
ENV_CREATED=false
if [ ! -f .env ]; then
    cp .env.example .env
    ENV_CREATED=true
    ok ".env created from .env.example"
fi

if $IS_WSL; then
    if [ "$(env_value AI_MODELS_ROOT)" = "/docker/ai-models" ]; then
        set_env_value AI_MODELS_ROOT "${HOME}/.local/share/ocabra/ai-models"
    fi
    if [ -z "$(env_value ACESTEP_ROOT)" ] || [ "$(env_value ACESTEP_ROOT)" = "/docker/ACE-Step-1.5" ]; then
        set_env_value ACESTEP_ROOT "${HOME}/.local/share/ocabra/ACE-Step-1.5"
    fi
    set_env_value THERMAL_CAP_ENABLED false
    set_env_value GPU_THROTTLE_ENABLED false
fi

[ -n "$(env_value JWT_SECRET)" ] || set_env_value JWT_SECRET "$(random_hex 32)"
[ -n "$(env_value GATEWAY_SERVICE_TOKEN)" ] || set_env_value GATEWAY_SERVICE_TOKEN "$(random_hex 32)"

if $ENV_CREATED && [ -t 0 ]; then
    read -rp "$(echo -e "${CYAN}Admin username${NC} [ocabra]: ")" admin_user
    set_env_value OCABRA_ADMIN_USER "${admin_user:-ocabra}"
    read -rsp "$(echo -e "${CYAN}Admin password${NC} [generated]: ")" admin_pass
    echo
    set_env_value OCABRA_ADMIN_PASS "${admin_pass:-$(random_hex 16)}"
fi

POSTGRES_USER="$(env_value POSTGRES_USER)"; POSTGRES_USER="${POSTGRES_USER:-ocabra}"
POSTGRES_DB="$(env_value POSTGRES_DB)"; POSTGRES_DB="${POSTGRES_DB:-ocabra}"
POSTGRES_PASSWORD="$(env_value POSTGRES_PASSWORD)"
if [ -z "$POSTGRES_PASSWORD" ] || [ "$POSTGRES_PASSWORD" = "change_me_in_production" ]; then
    POSTGRES_PASSWORD="$(random_hex 24)"
    set_env_value POSTGRES_PASSWORD "$POSTGRES_PASSWORD"
fi
ENCODED_POSTGRES_PASSWORD="$(urlencode "$POSTGRES_PASSWORD")"
set_env_value DATABASE_URL \
    "postgresql+asyncpg://${POSTGRES_USER}:${ENCODED_POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}"

AI_MODELS_ROOT="$(env_value AI_MODELS_ROOT)"; AI_MODELS_ROOT="${AI_MODELS_ROOT:-/docker/ai-models}"
ACESTEP_ROOT="$(env_value ACESTEP_ROOT)"; ACESTEP_ROOT="${ACESTEP_ROOT:-/docker/ACE-Step-1.5}"

# The api runs as this uid (not root, since v0.106.2) and must own what it
# writes: the folders below, created by this script as the current user.
if [ -z "$(env_value OCABRA_UID)" ]; then
    set_env_value OCABRA_UID "$(id -u)"
    set_env_value OCABRA_GID "$(id -g)"
fi
OCABRA_UID="$(env_value OCABRA_UID)"

info "Creating data directories..."
dirs=(
    ./data/postgres_data ./data/openai_files ./data/image_outputs ./data/ollama
    ./caddy/caddy_data ./caddy/caddy_config
    "$AI_MODELS_ROOT/ocabra/models" "$AI_MODELS_ROOT/ocabra/hf_cache"
    "$AI_MODELS_ROOT/ollama" "$ACESTEP_ROOT/checkpoints"
)
for directory in "${dirs[@]}"; do
    mkdir -p "$directory" || fatal "Cannot create $directory; choose writable AI_MODELS_ROOT/ACESTEP_ROOT paths in .env."
done
foreign=()
for directory in "${dirs[@]}"; do
    case "$directory" in ./data/postgres_data|./caddy/*|"$AI_MODELS_ROOT/ollama") continue ;; esac
    [ "$(stat -c %u "$directory")" = "$OCABRA_UID" ] || foreign+=("$directory")
done
if [ "${#foreign[@]}" -gt 0 ]; then
    warn "Not owned by uid $OCABRA_UID, the api cannot write there: ${foreign[*]}"
    warn "Fix it with: backend/scripts/chown-api-data.sh   (after the first build)"
fi
ok "Data directories ready"

"${COMPOSE[@]}" config -q || fatal "docker-compose.yml is not valid with the current .env."

CORE_SERVICES=(api docker-agent frontend postgres redis hw-monitor ollama caddy)
BUILD_SERVICES=(api frontend hw-monitor ollama)
START_SERVICES=("${CORE_SERVICES[@]}")

if $WITH_SERVICES; then
    START_SERVICES+=(gateway unsloth-studio)
    BUILD_SERVICES+=(gateway)
    if [ -d ./TRELLIS.2 ]; then START_SERVICES+=(trellis2); BUILD_SERVICES+=(trellis2); fi
    if [ -d ./services/comfyui ]; then START_SERVICES+=(comfyui); BUILD_SERVICES+=(comfyui); fi
    if [ -d ./services/a1111 ]; then START_SERVICES+=(a1111); BUILD_SERVICES+=(a1111); fi
    if [ -f "$ACESTEP_ROOT/Dockerfile" ]; then START_SERVICES+=(acestep); BUILD_SERVICES+=(acestep); fi
fi

info "Building Docker images: ${BUILD_SERVICES[*]}"
"${COMPOSE[@]}" build "${BUILD_SERVICES[@]}"
ok "Docker images built"

info "Starting oCabra services: ${START_SERVICES[*]}"
"${COMPOSE[@]}" up -d "${START_SERVICES[@]}"
ok "Containers started"

info "Waiting for API to become healthy..."
MAX_WAIT=120; ELAPSED=0
while [ "$ELAPSED" -lt "$MAX_WAIT" ]; do
    if curl -sf http://localhost:8484/health >/dev/null 2>&1; then break; fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
    printf '.'
done
echo

if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    err "API did not become healthy within ${MAX_WAIT}s."
    "${COMPOSE[@]}" ps
    fatal "Inspect logs with: docker compose logs api hw-monitor"
fi
ok "API is healthy"

if $IS_WSL; then
    warn "WSL2 GPU inference is enabled, but power limits and thermal caps are disabled."
    warn "Validate GPU ordering before relying on per-GPU placement on a multi-GPU Windows host."
fi

echo -e "\n${BOLD}=== oCabra is running ===${NC}\n"
echo -e "  Dashboard:  ${GREEN}http://localhost:8484${NC}"
echo -e "  OpenAI API: ${GREEN}http://localhost:8484/v1${NC}"
echo -e "  Ollama API: ${GREEN}http://localhost:8484/api${NC}"
echo -e "  API docs:   ${GREEN}http://localhost:8484/docs${NC}\n"
echo -e "  ${BOLD}docker compose logs -f api${NC}   Follow API logs"
echo -e "  ${BOLD}docker compose ps${NC}            Show running services"
echo -e "  ${BOLD}docker compose down${NC}          Stop all services\n"
