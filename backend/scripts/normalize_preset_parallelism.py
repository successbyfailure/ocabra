"""One-off: alinear parallelism con lo que el daemon Ollama reserva de verdad.

El ``parallelism`` del preset solo tiene efecto en el cálculo del vram_planner
y en la selección de pool. El daemon Ollama ignora el valor por request y
usa el env ``OLLAMA_NUM_PARALLEL`` del contenedor para reservar
``num_ctx × num_parallel`` tokens de KV cache. Fingir ``parallelism=1`` en
un pool con ``num_parallel=2`` no reduce el KV reservado — solo esconde el
problema del resolver, que entonces reporta ``fits=True`` cuando en
realidad el daemon reservará el doble y Ollama cae con CUDA OOM
(precisamente el fallo que vio rogelio con gemma4:26b 128k).

Alinea parallelism con el ``num_parallel`` del pool al que cada preset va
destinado por su ``preferred_gpus``:

- Presets ``32k`` y ``64k`` (destino 3090 np=2): parallelism=2.
- Presets ``128k`` (destino shared-np2 np=2): parallelism=2.
- Presets ``max`` de LLMs grandes (destino shared np=1): parallelism=1.
- Presets ``max`` de LLMs pequeños (destino 3090 np=1 virtual): parallelism=1
  para que el ``max_context_tokens`` del profile sea alcanzable.

Idempotente.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from ocabra.database import AsyncSessionLocal


LLM_LARGE = (
    "gemma4:26b",
    "qwen3.8:27b",
    "qwen3.8:27b-abliterated",
    "qwen3-coder:30b",
    "nemotron-3-nano:30b",
)
LLM_SMALL = ("gemma4:12b", "qwen3-vl:8b", "qwen3:8b")


async def main() -> None:
    async with AsyncSessionLocal() as session:
        touched = 0
        r = await session.execute(
            text(
                """
                UPDATE profile_runtime_presets
                SET parallelism=2, updated_at=now()
                WHERE preset_key IN ('32k','64k','128k')
                  AND profile_id = ANY(:ids)
                  AND parallelism != 2
                RETURNING profile_id, preset_key
                """
            ),
            {"ids": list(LLM_LARGE)},
        )
        touched += r.rowcount or 0
        r = await session.execute(
            text(
                """
                UPDATE profile_runtime_presets
                SET parallelism=1, updated_at=now()
                WHERE preset_key='max'
                  AND profile_id = ANY(:ids)
                  AND parallelism != 1
                RETURNING profile_id
                """
            ),
            {"ids": list(LLM_SMALL + LLM_LARGE)},
        )
        touched += r.rowcount or 0
        await session.commit()
        print(f"Normalized parallelism on {touched} presets.")


if __name__ == "__main__":
    asyncio.run(main())
