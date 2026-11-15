#!/usr/bin/env bash
# Entry point of the ocabra-node image (Dockerfile.node): start the private
# Postgres and Redis, then hand over to the normal entrypoint (migrations +
# API). If either service dies later the API fails its requests and the hub
# sees the node offline; the hub's watchdog and the node's self-destruct deal
# with a pod that stays broken.
set -euo pipefail

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

echo "[ocabra-node] postgres and redis up; starting oCabra (joining ${FEDERATION_JOIN_URL:-<no hub>})"
exec /app/docker-entrypoint.sh "$@"
