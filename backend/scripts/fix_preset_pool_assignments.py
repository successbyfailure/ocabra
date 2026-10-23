"""One-off: re-route oversized presets to the shared pool and cap parallelism.

Audited 2026-10-23 using ``runtime-presets/resolved`` across every public
LLM profile. Found presets whose ``context_tokens`` did not fit the
assigned pool with the configured ``parallelism`` (Ollama reserves
``num_ctx × num_parallel`` KV tokens up front, which multiplied the real
footprint):

- ``nemotron-3-nano:30b`` (MoE 31B, 18 GB weights): every ``fixed`` preset
  (32k/64k/128k) was clamped by vram_planner to ≤36k. 3090 alone has no
  budget for anything beyond a tiny context — belongs in ``shared``.
- ``qwen3-coder:30b`` (MoE 30B, 18 GB): 64k clamped to 62k, 128k to 124k.
- ``qwen3.8:27b-abliterated`` (27B Q5, 21 GB): 128k clamped to 97k.
- ``qwen3-vl:8b`` (8B + ViT): 128k clamped to 117k, needed np=1.
- ``gemma4:26b`` and ``qwen3.8:27b`` 64k warm defaults were marked
  fits=True but at the very edge of 24 GB (~24.8 GB total); lowered their
  parallelism from 2 to 1 so an unexpected spike doesn't push them
  into CUDA OOM territory (same failure mode that bit rogelio on 128k).

Resulting placements after this patch:

                       32k    64k    128k   max
    gemma4:26b         3090   3090   3090   shared
    qwen3.8:27b        3090   3090   3090   shared
    qwen3.8:27b-abli   3090   3090   shared shared
    qwen3-coder:30b    auto   shared shared shared
    nemotron-3-nano    shared shared shared shared
    qwen3-vl:8b        3090   3090   3090   3090

Safe to re-run: idempotent (only touches presets whose current placement
differs from the target).
"""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from ocabra.database import AsyncSessionLocal


UPDATES = [
    # (profile_id, preset_keys, placement_policy, preferred_gpus_jsonb, parallelism)
    ("nemotron-3-nano:30b", ("32k", "64k", "128k"), "forced", "[0,1]", 1),
    ("qwen3-coder:30b",     ("64k", "128k"),         "forced", "[0,1]", 1),
    ("qwen3.8:27b-abliterated", ("128k",),           "forced", "[0,1]", 1),
    ("qwen3-vl:8b",         ("128k",),               None,     None,    1),
    ("gemma4:26b",          ("64k",),                None,     None,    1),
    ("qwen3.8:27b",         ("64k",),                None,     None,    1),
]


async def main() -> None:
    async with AsyncSessionLocal() as session:
        for profile_id, keys, placement, gpus, parallelism in UPDATES:
            params = {"profile_id": profile_id, "keys": list(keys), "p": parallelism}
            if placement is None:
                sql = """
                    UPDATE profile_runtime_presets
                    SET parallelism=:p, updated_at=now()
                    WHERE profile_id=:profile_id
                      AND preset_key = ANY(:keys)
                      AND parallelism != :p
                    RETURNING preset_key
                """
            else:
                sql = """
                    UPDATE profile_runtime_presets
                    SET placement_policy=:placement,
                        preferred_gpus=CAST(:gpus AS jsonb),
                        parallelism=:p,
                        updated_at=now()
                    WHERE profile_id=:profile_id
                      AND preset_key = ANY(:keys)
                      AND (placement_policy != :placement
                           OR preferred_gpus != CAST(:gpus AS jsonb)
                           OR parallelism != :p)
                    RETURNING preset_key
                """
                params["placement"] = placement
                params["gpus"] = gpus
            result = await session.execute(text(sql), params)
            touched = [row[0] for row in result]
            if touched:
                print(f"{profile_id}: {touched}")
        await session.commit()


if __name__ == "__main__":
    asyncio.run(main())
