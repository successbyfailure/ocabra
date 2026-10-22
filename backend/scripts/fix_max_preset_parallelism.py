"""One-off: force parallelism=1 on multi-GPU ``max`` presets.

Pool ``ollama-shared`` runs with ``OLLAMA_NUM_PARALLEL=1`` because 2 slots
× max context would reserve more KV cache than fits (observed CUDA OOM on
2026-10-21). ``_select_pool`` rejects a load whose preset ``parallelism``
exceeds the pool's ``num_parallel``, so any preset still carrying the
earlier ``parallelism=2`` from the first migration cannot be satisfied by
the shared pool and the client sees ``no Ollama pool exposes GPUs [0, 1]
with multi_gpu=true and parallelism 2`` (observed on nemotron-3-nano:30b
max and qwen3-coder:30b max, 2026-10-22 11:28/11:34 UTC).

Safe to re-run: idempotent.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from ocabra.database import AsyncSessionLocal


async def main() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                """
                UPDATE profile_runtime_presets
                SET parallelism=1, updated_at=now()
                WHERE preset_key='max'
                  AND placement_policy='forced'
                  AND preferred_gpus @> '[0,1]'::jsonb
                  AND parallelism > 1
                RETURNING profile_id
                """
            )
        )
        fixed = [row[0] for row in result]
        await session.commit()
        if fixed:
            print(f"Reset parallelism=1 for max preset on: {fixed}")
        else:
            print("Nothing to do; all multi-GPU max presets already parallelism=1.")


if __name__ == "__main__":
    asyncio.run(main())
