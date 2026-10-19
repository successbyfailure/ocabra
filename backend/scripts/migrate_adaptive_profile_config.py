"""Normalize the production profile inventory for adaptive context presets.

Dry-run is the default. Pass ``--apply`` after Alembic revision 0026 is active.
The operation is idempotent and leaves hidden aliases for every retired public
``-ctx*`` identifier; it does not delete Ollama blobs from disk.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from ocabra.database import AsyncSessionLocal


LEGACY_PROFILES = (
    "gemma4:12b-ctx256k",
    "gemma4:26b-ctx160k",
    "gemma4:26b-ollama",
    "qwen3.8:27b-ctx160k",
    "qwen3.8:27b-abliterated-ctx160k",
)

LEGACY_MODELS = (
    "ollama/gemma4:12b-ctx256k",
    "ollama/gemma4:26b-ctx160k",
    "ollama/qwen3.8:27b-ctx160k",
    "ollama/qwen3.8:27b-abliterated-ctx160k",
)


async def _inventory(session) -> None:
    rows = (
        await session.execute(
            text(
                """
                SELECT profile_id, profile_kind, base_model_id, routing_targets
                FROM model_profiles
                WHERE profile_id = ANY(:ids)
                   OR profile_id LIKE 'recomendado-%'
                ORDER BY profile_id
                """
            ),
            {"ids": list(LEGACY_PROFILES)},
        )
    ).mappings()
    for row in rows:
        print(dict(row))


async def migrate(apply: bool) -> None:
    async with AsyncSessionLocal() as session:
        print("Current adaptive-profile inventory:")
        await _inventory(session)
        if not apply:
            print("Dry-run only; rerun with --apply to persist the cleanup.")
            return

        # ``AsyncSessionLocal()`` opens a transaction automatically on first
        # use, so we commit at the end rather than nesting another begin().
        try:
            await session.execute(
                text(
                    """
                    UPDATE model_profiles
                    SET display_name = 'Gemma 4 26B',
                        description = 'Gemma 4 26B Ollama con contexto adaptativo.',
                        is_default = TRUE,
                        updated_at = now()
                    WHERE profile_id = 'gemma4:26b'
                    """
                )
            )
            await session.execute(
                text(
                    """
                    UPDATE model_profiles
                    SET routing_targets = CAST(:targets AS jsonb),
                        description = :description,
                        updated_at = now()
                    WHERE profile_id = :profile_id
                    """
                ),
                [
                    {
                        "profile_id": "recomendado-ha-chatbots",
                        "targets": '["gemma4:26b","qwen3.8:27b","qwen3.8:27b-abliterated","gemma4:12b"]',
                        "description": "Alta disponibilidad: reutiliza el mejor perfil LLM disponible sin fijar contexto.",
                    },
                    {
                        "profile_id": "recomendado-sota",
                        "targets": '["qwen3.8:27b","qwen3.8:27b-abliterated","gemma4:26b"]',
                        "description": "Calidad prioritaria con selección adaptativa de contexto y fallback ordenado.",
                    },
                    {
                        "profile_id": "recomendado-stt",
                        "targets": '["faster-whisper-large-v3"]',
                        "description": "Transcripción recomendada con fallback administrado por router.",
                    },
                ],
            )
            await session.execute(
                text(
                    """
                    UPDATE model_profiles
                    SET category = 'stt', updated_at = now()
                    WHERE profile_id = 'recomendado-stt'
                    """
                )
            )

            # Qwen 3.8 currently requests p2 for the normal presets, but
            # Ollama/qwen35 enforces p1. The planner consumes this cap before
            # sizing KV and the UI shows requested versus effective values.
            # Use jsonb_build_object instead of a literal '{"a":1}' so
            # SQLAlchemy doesn't treat the ``:`` inside the JSON as bind
            # parameter markers (``:1`` reads as placeholder "1").
            await session.execute(
                text(
                    """
                    UPDATE model_configs
                    SET extra_config = jsonb_set(
                        COALESCE(extra_config, '{}'::jsonb),
                        '{runtime_capabilities}',
                        jsonb_build_object(
                            'max_parallelism', 1,
                            'reason', 'ollama_qwen35_single_sequence'
                        ),
                        true
                    ), updated_at = now()
                    WHERE model_id IN (
                        'ollama/qwen3.8:27b',
                        'ollama/qwen3.8:27b-abliterated'
                    )
                    """
                )
            )
            await session.execute(
                text(
                    """
                    UPDATE model_profiles
                    SET runtime_policy = COALESCE(runtime_policy, '{}'::jsonb) || jsonb_build_object(
                            'selection', 'prefer_loaded',
                            'max_context_tokens', 163840,
                            'restore_default_when_idle', true,
                            'downgrade_grace_seconds', 600,
                            'minimum_residency_seconds', 120,
                            'small_requests_extend_large_ttl', false,
                            'remote_warm_threshold_seconds', 10
                        ),
                        updated_at = now()
                    WHERE profile_id IN ('qwen3.8:27b', 'qwen3.8:27b-abliterated')
                    """
                )
            )

            await session.execute(
                text(
                    """
                    INSERT INTO profile_runtime_presets (
                        id, profile_id, preset_key, label, context_mode,
                        context_tokens, parallelism, kv_cache_dtype,
                        placement_policy, preferred_gpus, execution_policy,
                        load_policy, enabled, is_default, priority
                    ) VALUES
                        (gen_random_uuid(), 'qwen3.8:27b-abliterated', '32k', '32K', 'fixed', 32768, 1, 'q8', 'forced', '[1]'::jsonb, 'auto', 'on_demand', TRUE, FALSE, 10),
                        (gen_random_uuid(), 'qwen3.8:27b-abliterated', '64k', '64K', 'fixed', 65536, 1, 'q8', 'forced', '[1]'::jsonb, 'auto', 'warm', TRUE, TRUE, 20),
                        (gen_random_uuid(), 'qwen3.8:27b-abliterated', '128k', '128K', 'fixed', 131072, 1, 'q8', 'forced', '[1]'::jsonb, 'auto', 'on_demand', TRUE, FALSE, 30),
                        (gen_random_uuid(), 'qwen3.8:27b-abliterated', 'max', 'MAX', 'max', NULL, 1, 'q8', 'forced', '[1]'::jsonb, 'auto', 'on_demand', TRUE, FALSE, 40)
                    ON CONFLICT (profile_id, preset_key) DO NOTHING
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO model_profile_aliases (
                        alias_id, profile_id, preset_key, hidden
                    ) VALUES (
                        'qwen3.8:27b-abliterated-ctx160k',
                        'qwen3.8:27b-abliterated',
                        'max',
                        TRUE
                    ) ON CONFLICT (alias_id) DO UPDATE SET
                        profile_id = EXCLUDED.profile_id,
                        preset_key = EXCLUDED.preset_key,
                        hidden = TRUE
                    """
                )
            )

            # Agents must consume the public adaptive profile rather than the
            # raw model config, otherwise they bypass preset selection.
            await session.execute(
                text(
                    """
                    UPDATE agents
                    SET profile_id = 'gemma4:26b', base_model_id = NULL,
                        updated_at = now()
                    WHERE base_model_id = 'ollama/gemma4:26b'
                    """
                )
            )

            # Preserve existing access while removing permissions for retired
            # physical tags. Router permission is explicit now that routers do
            # not inherit a fictitious base model.
            await session.execute(
                text(
                    """
                    INSERT INTO group_models (group_id, model_id)
                    SELECT DISTINCT gm.group_id, router.profile_id
                    FROM group_models AS gm
                    CROSS JOIN (VALUES
                        ('recomendado-ha-chatbots', 'ollama/gemma4:26b'),
                        ('recomendado-sota', 'ollama/qwen3.8:27b'),
                        ('recomendado-stt', 'whisper/Systran/faster-whisper-large-v3')
                    ) AS router(profile_id, required_model_id)
                    WHERE gm.model_id = router.required_model_id
                    ON CONFLICT DO NOTHING
                    """
                )
            )
            await session.execute(
                text("DELETE FROM group_models WHERE model_id = ANY(:ids)"),
                {"ids": list(LEGACY_MODELS)},
            )

            await session.execute(
                text("DELETE FROM model_profiles WHERE profile_id = ANY(:ids)"),
                {"ids": list(LEGACY_PROFILES)},
            )
            await session.execute(
                text("DELETE FROM model_configs WHERE model_id = ANY(:ids)"),
                {"ids": list(LEGACY_MODELS)},
            )
            await session.commit()
        except Exception:
            await session.rollback()
            raise

        print("Cleanup applied. Resulting inventory:")
        await _inventory(session)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    asyncio.run(migrate(args.apply))


if __name__ == "__main__":
    main()
