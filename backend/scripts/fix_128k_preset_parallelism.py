"""One-off: force parallelism=1 on 128k presets for large LLMs.

Ollama allocates ``num_ctx × num_parallel`` tokens of KV cache up front.
With parallelism=2 on the ``128k`` preset of gemma4:26b (and the other
large LLMs forced to a single 3090), Ollama reserved ``2 × 131072 =
262144`` tokens of KV cache in a single 24 GB GPU — ~15 GB of KV +
~17 GB of weights = ~32 GB, which overflows the 3090 and causes the
llama-server to crash with CUDA OOM mid-tokenize. Clients see
``Post "http://127.0.0.1:XXXXX/tokenize": EOF`` after ~7 minutes
(reported by rogeliomarquezgarcia 2026-10-23 16:10-16:40 UTC with a
60k-token prompt that previously fit on ``gemma4:26b-ctx160k``).

Lowering parallelism=1 halves the reserved KV budget so 131072 tokens
really fit in 24 GB. For single-slot concurrency it means two
simultaneous 128k requests serialize instead of fighting over VRAM.

Safe to re-run: idempotent.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from ocabra.database import AsyncSessionLocal


LARGE_LLM_PROFILES = (
    "gemma4:26b",
    "qwen3.8:27b",
    "qwen3.8:27b-abliterated",
    "qwen3-coder:30b",
    "nemotron-3-nano:30b",
)


async def main() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                """
                UPDATE profile_runtime_presets
                SET parallelism=1, updated_at=now()
                WHERE preset_key='128k'
                  AND profile_id = ANY(:ids)
                  AND parallelism > 1
                RETURNING profile_id
                """
            ),
            {"ids": list(LARGE_LLM_PROFILES)},
        )
        fixed = [row[0] for row in result]
        await session.commit()
        if fixed:
            print(f"Reset parallelism=1 for 128k preset on: {fixed}")
        else:
            print("Nothing to do; all 128k presets already parallelism=1.")


if __name__ == "__main__":
    asyncio.run(main())
