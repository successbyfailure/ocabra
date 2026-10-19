"""Second pass: migrate nemotron-3-nano:30b and qwen3-coder:30b to adaptive presets.

Mirrors ``migrate_adaptive_profile_config.py`` but for the two Ollama profiles
that were left behind in the first pass. Both ``-ctx256k`` variants only differ
from their canonical profile in context length, so they fold cleanly into
preset+alias with no new runtime shape.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from ocabra.database import AsyncSessionLocal


LEGACY_PROFILES = (
    "nemotron-3-nano:30b-ctx256k",
    "qwen3-coder:30b-ctx256k",
)

LEGACY_MODELS = (
    "ollama/nemotron-3-nano:30b-ctx256k",
    "ollama/qwen3-coder:30b-ctx256k",
)

CANONICAL_PROFILES = (
    "nemotron-3-nano:30b",
    "qwen3-coder:30b",
)

CANONICAL_MODELS = (
    "ollama/nemotron-3-nano:30b",
    "ollama/qwen3-coder:30b",
)


async def _inventory(session) -> None:
    rows = (
        await session.execute(
            text(
                """
                SELECT profile_id, profile_kind, base_model_id, is_default
                FROM model_profiles
                WHERE profile_id = ANY(:ids)
                ORDER BY profile_id
                """
            ),
            {"ids": list(LEGACY_PROFILES + CANONICAL_PROFILES)},
        )
    ).mappings()
    for row in rows:
        print(dict(row))


async def migrate(apply: bool) -> None:
    async with AsyncSessionLocal() as session:
        print("Inventory before:")
        await _inventory(session)
        if not apply:
            print("Dry-run only; rerun with --apply to persist the cleanup.")
            return

        try:
            # Enrich canonical profiles so the preset UI shows sensible metadata.
            await session.execute(
                text(
                    """
                    UPDATE model_profiles
                    SET display_name = 'Nemotron 3 Nano 30B',
                        description = 'Nemotron 3 Nano 30B (Ollama) con contexto adaptativo.',
                        updated_at = now()
                    WHERE profile_id = 'nemotron-3-nano:30b'
                    """
                )
            )
            await session.execute(
                text(
                    """
                    UPDATE model_profiles
                    SET display_name = 'Qwen3 Coder 30B',
                        description = 'Qwen3 Coder 30B (Ollama) con contexto adaptativo.',
                        updated_at = now()
                    WHERE profile_id = 'qwen3-coder:30b'
                    """
                )
            )

            # Runtime policy mirrors the one used for gemma4:26b and qwen3.8:27b
            # so planner behaviour is uniform across the public LLM profiles.
            await session.execute(
                text(
                    """
                    UPDATE model_profiles
                    SET runtime_policy = COALESCE(runtime_policy, '{}'::jsonb) || jsonb_build_object(
                            'selection', 'prefer_loaded',
                            'max_context_tokens', 262144,
                            'restore_default_when_idle', true,
                            'downgrade_grace_seconds', 600,
                            'minimum_residency_seconds', 120,
                            'small_requests_extend_large_ttl', false,
                            'remote_warm_threshold_seconds', 10
                        ),
                        updated_at = now()
                    WHERE profile_id = ANY(:ids)
                    """
                ),
                {"ids": list(CANONICAL_PROFILES)},
            )

            # Four presets each: 32K / 64K (default, warm) / 128K / MAX (256K).
            await session.execute(
                text(
                    """
                    INSERT INTO profile_runtime_presets (
                        id, profile_id, preset_key, label, context_mode,
                        context_tokens, parallelism, kv_cache_dtype,
                        placement_policy, preferred_gpus, execution_policy,
                        load_policy, enabled, is_default, priority
                    ) VALUES
                        (gen_random_uuid(), 'nemotron-3-nano:30b', '32k', '32K', 'fixed', 32768, 2, 'q8', 'auto', NULL, 'auto', 'on_demand', TRUE, FALSE, 10),
                        (gen_random_uuid(), 'nemotron-3-nano:30b', '64k', '64K', 'fixed', 65536, 2, 'q8', 'auto', NULL, 'auto', 'warm', TRUE, TRUE, 20),
                        (gen_random_uuid(), 'nemotron-3-nano:30b', '128k', '128K', 'fixed', 131072, 2, 'q8', 'auto', NULL, 'auto', 'on_demand', TRUE, FALSE, 30),
                        (gen_random_uuid(), 'nemotron-3-nano:30b', 'max', 'MAX', 'max', NULL, 2, 'q8', 'auto', NULL, 'auto', 'on_demand', TRUE, FALSE, 40),
                        (gen_random_uuid(), 'qwen3-coder:30b', '32k', '32K', 'fixed', 32768, 2, 'q8', 'auto', NULL, 'auto', 'on_demand', TRUE, FALSE, 10),
                        (gen_random_uuid(), 'qwen3-coder:30b', '64k', '64K', 'fixed', 65536, 2, 'q8', 'auto', NULL, 'auto', 'warm', TRUE, TRUE, 20),
                        (gen_random_uuid(), 'qwen3-coder:30b', '128k', '128K', 'fixed', 131072, 2, 'q8', 'auto', NULL, 'auto', 'on_demand', TRUE, FALSE, 30),
                        (gen_random_uuid(), 'qwen3-coder:30b', 'max', 'MAX', 'max', NULL, 2, 'q8', 'auto', NULL, 'auto', 'on_demand', TRUE, FALSE, 40)
                    ON CONFLICT (profile_id, preset_key) DO NOTHING
                    """
                )
            )

            # Hidden aliases keep the old public identifiers resolving.
            await session.execute(
                text(
                    """
                    INSERT INTO model_profile_aliases (
                        alias_id, profile_id, preset_key, hidden
                    ) VALUES
                        ('nemotron-3-nano:30b-ctx256k', 'nemotron-3-nano:30b', 'max', TRUE),
                        ('qwen3-coder:30b-ctx256k', 'qwen3-coder:30b', 'max', TRUE)
                    ON CONFLICT (alias_id) DO UPDATE SET
                        profile_id = EXCLUDED.profile_id,
                        preset_key = EXCLUDED.preset_key,
                        hidden = TRUE
                    """
                )
            )

            # Preserve group permissions: whoever had the legacy ollama/*-ctx256k
            # or the base ollama/* keeps access via the canonical profile_id.
            await session.execute(
                text(
                    """
                    INSERT INTO group_models (group_id, model_id)
                    SELECT DISTINCT gm.group_id, mapping.profile_id
                    FROM group_models AS gm
                    CROSS JOIN (VALUES
                        ('nemotron-3-nano:30b', 'ollama/nemotron-3-nano:30b'),
                        ('nemotron-3-nano:30b', 'ollama/nemotron-3-nano:30b-ctx256k'),
                        ('qwen3-coder:30b', 'ollama/qwen3-coder:30b'),
                        ('qwen3-coder:30b', 'ollama/qwen3-coder:30b-ctx256k')
                    ) AS mapping(profile_id, required_model_id)
                    WHERE gm.model_id = mapping.required_model_id
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

        print("Cleanup applied. Inventory after:")
        await _inventory(session)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    asyncio.run(migrate(args.apply))


if __name__ == "__main__":
    main()
