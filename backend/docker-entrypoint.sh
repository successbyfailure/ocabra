#!/usr/bin/env bash
set -euo pipefail

wait_for_postgres() {
    python - <<'PY'
import asyncio
import sys

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from ocabra.config import settings


async def main() -> int:
    url = make_url(settings.database_url)
    attempts = 30

    for attempt in range(1, attempts + 1):
        try:
            engine = create_async_engine(settings.database_url, pool_pre_ping=True)
            async with engine.connect() as connection:
                await connection.exec_driver_sql("SELECT 1")
            await engine.dispose()
            print(
                f"Postgres ready at {url.host}:{url.port}/{url.database} "
                f"after {attempt} attempt(s)."
            )
            return 0
        except Exception as exc:
            print(f"Waiting for Postgres ({attempt}/{attempts}): {exc}", file=sys.stderr)
            await asyncio.sleep(1)

    return 1


raise SystemExit(asyncio.run(main()))
PY
}

should_run_migrations() {
    if [ "${OCABRA_SKIP_MIGRATIONS:-false}" = "true" ]; then
        return 1
    fi

    if [ "$#" -eq 0 ]; then
        return 0
    fi

    [ "$1" = "uvicorn" ]
}

if should_run_migrations "$@"; then
    wait_for_postgres
    alembic upgrade head
fi

exec "$@"
