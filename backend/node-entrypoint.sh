#!/usr/bin/env bash
# Entry point of the ocabra-node image (Dockerfile.node): start the private
# Postgres and Redis, then hand over to the normal entrypoint (migrations +
# API). If either service dies later the API fails its requests and the hub
# sees the node offline; the hub's watchdog and the node's self-destruct deal
# with a pod that stays broken.
set -euo pipefail

# No core dumps: a crashing worker must not leave its memory (prompts,
# outputs) on the disk of a machine we don't control.
ulimit -c 0 || true

# Data dirs the API expects (models, HF cache, Ollama store) exist from the
# start, not only after the first download.
mkdir -p /data/models /data/hf_cache /data/ollama_models

# Network volume (OCABRA_PERSIST_DIR, mounted by the hub when it launches the
# pod with one): models, HF cache and Ollama store live there, so a new pod
# finds them already downloaded. Backends do NOT: a venv is tens of thousands
# of small files, and the network volume pays a round trip per file — the
# first pod with a volume spent over 10 minutes just unpacking diffusers there
# (35 s on a local disk), and every import would pay it again. They install
# prebuilt from GHCR to the local disk in ~90 s instead. The database, Redis,
# batches and outputs stay on the ephemeral disk too.
if [ -n "${OCABRA_PERSIST_DIR:-}" ] && [ -d "${OCABRA_PERSIST_DIR}" ]; then
    P="${OCABRA_PERSIST_DIR}"
    mkdir -p "$P/models" "$P/hf_cache" "$P/ollama_models"
    for d in models hf_cache ollama_models; do
        if [ -d "/data/$d" ] && [ ! -L "/data/$d" ]; then
            cp -an "/data/$d/." "$P/$d/" 2>/dev/null || true
            rm -rf "/data/$d"
        fi
        ln -sfn "$P/$d" "/data/$d"
    done
    echo "[ocabra-node] models on the network volume at $P"
fi

PG_BIN="$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -1)"
PGDATA=/var/lib/postgresql/data

if [ ! -s "${PGDATA}/PG_VERSION" ]; then
    mkdir -p "${PGDATA}"
    chown -R postgres:postgres /var/lib/postgresql
    su postgres -c "${PG_BIN}/initdb -D ${PGDATA} -U postgres --auth=trust" >/dev/null
fi
su postgres -c "${PG_BIN}/pg_ctl -D ${PGDATA} -l /var/lib/postgresql/server.log \
    -o \"-c listen_addresses=127.0.0.1\" -w start" >/dev/null
su postgres -c "psql -qtAc \"SELECT 1 FROM pg_roles WHERE rolname='ocabra'\"" | grep -q 1 \
    || su postgres -c "psql -qc \"CREATE ROLE ocabra LOGIN PASSWORD 'ocabra'\""
su postgres -c "psql -qtAc \"SELECT 1 FROM pg_database WHERE datname='ocabra'\"" | grep -q 1 \
    || su postgres -c "createdb -O ocabra ocabra"

redis-server --bind 127.0.0.1 --port 6379 --save "" --appendonly no --daemonize yes >/dev/null

# The driver's libcuda must have been mounted by the NVIDIA runtime; without it
# every model would load "fine" and run on the CPU.
if ldconfig -p | grep -q 'libcuda\.so\.1'; then
    echo "[ocabra-node] CUDA driver library present"
else
    echo "[ocabra-node] WARNING: libcuda.so.1 not found: the GPU is not usable for compute here" \
         "(NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-unset}). Models will run on the CPU." >&2
fi
echo "[ocabra-node] postgres and redis up; starting oCabra (joining ${FEDERATION_JOIN_URL:-<no hub>})"
exec /app/docker-entrypoint.sh "$@"
