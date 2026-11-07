"""Dry-run of the model/backends unification (migration 0034).

Prints, read-only, exactly what migration 0034 would do on this database: the
logical models and their deployments, what happens to every profile / alias /
router / ACL entry / agent / stats row, and every anomaly found. ``--json``
dumps the raw plan.

Usage (inside the api container, where /data and the DB are reachable)::

    docker compose exec -T api python - < backend/scripts/model_unification_plan.py
    docker compose exec -T api python - --json < backend/scripts/model_unification_plan.py

Logic lives in ``ocabra.core.model_unification``; see
docs/tasks/model-backends-unification-plan.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def _plan() -> dict:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from ocabra.config import settings
    from ocabra.core.model_unification import build_plan, snapshot

    engine = create_async_engine(settings.database_url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SET TRANSACTION READ ONLY"))
            snap = await conn.run_sync(snapshot)
            await conn.rollback()
    finally:
        await engine.dispose()
    return build_plan(snap)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run of migration 0034 (model/backends unification)"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the plan as JSON instead of the report"
    )
    args = parser.parse_args()
    plan = asyncio.run(_plan())
    if args.json:
        json.dump(plan, sys.stdout, indent=2, ensure_ascii=False, default=str)
    else:
        from ocabra.core.model_unification import render

        print(render(plan))


if __name__ == "__main__":
    main()
