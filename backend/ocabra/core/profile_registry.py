"""In-memory registry for model profiles with DB-backed persistence."""

from __future__ import annotations

import hashlib
import re

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ocabra.db.model_config import (
    ModelConfig,
    ModelProfile,
    ModelProfileAlias,
    ProfileRuntimePreset,
)
from ocabra.schemas.profiles import RuntimePresetCreate

logger = structlog.get_logger(__name__)

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9.\-:]*$")
_MAX_PROFILE_ID_LEN = 512


def _is_valid_slug(value: str) -> bool:
    """Return True if value is a valid profile slug."""
    return bool(value) and len(value) <= _MAX_PROFILE_ID_LEN and _SLUG_RE.match(value) is not None


class ProfileRegistry:
    """In-memory cache of model profiles, backed by PostgreSQL.

    All mutations write-through to the DB and refresh the cache.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ModelProfile] = {}
        self._aliases: dict[str, ModelProfileAlias] = {}

    # ── Bulk load ────────────────────────────────────────────

    async def load_all(self, session: AsyncSession) -> None:
        """Load all profiles from the database into memory."""
        result = await session.execute(
            select(ModelProfile).options(
                selectinload(ModelProfile.runtime_presets),
                selectinload(ModelProfile.aliases),
            )
        )
        rows = result.scalars().all()
        self._profiles = {p.profile_id: p for p in rows}
        alias_result = await session.execute(select(ModelProfileAlias))
        aliases = alias_result.scalars().all()
        self._aliases = {alias.alias_id: alias for alias in aliases}
        logger.info("profile_registry_loaded", count=len(self._profiles))

    # ── Read operations (from cache) ─────────────────────────

    async def get(self, profile_id: str) -> ModelProfile | None:
        """Return a single profile by id, or None."""
        return self._profiles.get(profile_id)

    async def resolve_alias(
        self, alias_id: str
    ) -> tuple[ModelProfile, str | None] | None:
        alias = self._aliases.get(alias_id)
        if alias is None:
            return None
        profile = self._profiles.get(alias.profile_id)
        return (profile, alias.preset_key) if profile is not None else None

    async def create_alias(
        self,
        session: AsyncSession,
        *,
        alias_id: str,
        profile_id: str,
        preset_key: str | None = None,
        hidden: bool = True,
    ) -> ModelProfileAlias:
        if not _is_valid_slug(alias_id):
            raise ValueError(f"Invalid alias_id '{alias_id}'.")
        if alias_id in self._aliases:
            raise ValueError(f"Alias '{alias_id}' already exists.")
        if alias_id in self._profiles:
            raise ValueError(f"Alias '{alias_id}' conflicts with an existing profile.")
        if alias_id == profile_id:
            raise ValueError("An alias cannot target the profile with the same id.")
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ValueError(f"Profile '{profile_id}' not found.")
        if preset_key and not any(
            preset.preset_key == preset_key for preset in profile.runtime_presets
        ):
            raise ValueError(
                f"Runtime preset '{preset_key}' not found in profile '{profile_id}'."
            )
        alias = ModelProfileAlias(
            alias_id=alias_id,
            profile_id=profile_id,
            preset_key=preset_key,
            hidden=hidden,
        )
        session.add(alias)
        await session.commit()
        await session.refresh(alias)
        self._aliases[alias_id] = alias
        # ``profile`` is the long-lived cached instance in ``self._profiles``,
        # not the one ``session`` just touched — its ``aliases`` relationship
        # was eagerly loaded once at registry startup and never gets
        # invalidated by later writes through a different session. Without
        # this, a GET right after create/delete serves the stale list until
        # the next restart repopulates the cache from the DB.
        profile.aliases.append(alias)
        return alias

    async def delete_alias(self, session: AsyncSession, alias_id: str) -> None:
        alias = self._aliases.get(alias_id)
        if alias is None:
            raise ValueError(f"Alias '{alias_id}' not found.")
        await session.delete(alias)
        await session.commit()
        self._aliases.pop(alias_id, None)
        profile = self._profiles.get(alias.profile_id)
        if profile is not None:
            profile.aliases = [a for a in profile.aliases if a.alias_id != alias_id]

    async def list_by_model(self, base_model_id: str) -> list[ModelProfile]:
        """Return all profiles for a given base model."""
        return [p for p in self._profiles.values() if p.base_model_id == base_model_id]

    async def list_enabled(self) -> list[ModelProfile]:
        """Return all enabled profiles."""
        hidden_alias_ids = {
            alias.alias_id for alias in self._aliases.values() if alias.hidden
        }
        return [
            profile
            for profile in self._profiles.values()
            if profile.enabled and profile.profile_id not in hidden_alias_ids
        ]

    async def list_all(self) -> list[ModelProfile]:
        """Return all profiles."""
        return list(self._profiles.values())

    # ── Write operations (write-through) ─────────────────────

    async def create(
        self,
        session: AsyncSession,
        *,
        profile_id: str,
        base_model_id: str | None,
        profile_kind: str = "execution",
        display_name: str | None = None,
        description: str | None = None,
        category: str = "llm",
        load_overrides: dict | None = None,
        request_defaults: dict | None = None,
        assets: dict | None = None,
        enabled: bool = True,
        is_default: bool = False,
        routing_targets: list[str] | None = None,
        runtime_policy: dict | None = None,
    ) -> ModelProfile:
        """Create a new profile, validate constraints, persist, and update cache."""
        if not _is_valid_slug(profile_id):
            raise ValueError(
                f"Invalid profile_id '{profile_id}': must be lowercase alphanumeric "
                "with hyphens/dots, starting with an alphanumeric character."
            )
        if profile_id in self._profiles:
            raise ValueError(f"Profile '{profile_id}' already exists.")
        if profile_id in self._aliases:
            raise ValueError(f"Profile '{profile_id}' conflicts with a compatibility alias.")

        if category not in ("llm", "tts", "stt", "image", "music"):
            raise ValueError(f"Invalid category '{category}'.")

        if profile_kind not in {"execution", "router"}:
            raise ValueError(f"Invalid profile_kind '{profile_kind}'.")
        if profile_kind == "execution":
            if not base_model_id:
                raise ValueError("Execution profiles require a base_model_id.")
            if routing_targets:
                raise ValueError("Execution profiles cannot declare routing_targets.")
            result = await session.execute(
                select(ModelConfig).where(ModelConfig.model_id == base_model_id)
            )
            if result.scalar_one_or_none() is None:
                raise ValueError(f"Base model '{base_model_id}' not found.")
        else:
            if base_model_id is not None:
                raise ValueError("Router profiles cannot declare a base_model_id.")
            await self._validate_router_targets(profile_id, routing_targets)
            load_overrides = None
            runtime_policy = None
            is_default = False

        # If marking as default, clear existing default for this base model
        if is_default and base_model_id is not None:
            await self._clear_default(session, base_model_id)

        profile = ModelProfile(
            profile_id=profile_id,
            profile_kind=profile_kind,
            base_model_id=base_model_id,
            display_name=display_name,
            description=description,
            category=category,
            load_overrides=load_overrides,
            request_defaults=request_defaults,
            assets=assets,
            enabled=enabled,
            is_default=is_default,
            routing_targets=routing_targets,
            runtime_policy=runtime_policy,
        )
        session.add(profile)
        await session.commit()
        await session.refresh(profile)
        self._profiles[profile_id] = profile
        logger.info("profile_created", profile_id=profile_id, base_model_id=base_model_id)
        return profile

    async def update(
        self,
        session: AsyncSession,
        profile_id: str,
        patch: dict,
    ) -> ModelProfile:
        """Update an existing profile with a partial dict and refresh cache."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ValueError(f"Profile '{profile_id}' not found.")

        # Re-fetch attached to this session
        result = await session.execute(
            select(ModelProfile).where(ModelProfile.profile_id == profile_id)
        )
        db_profile = result.scalar_one_or_none()
        if db_profile is None:
            raise ValueError(f"Profile '{profile_id}' not found in database.")

        allowed_keys = {
            "display_name",
            "description",
            "category",
            "load_overrides",
            "request_defaults",
            "assets",
            "enabled",
            "is_default",
            "routing_targets",
            "runtime_policy",
        }
        if db_profile.profile_kind == "router":
            forbidden = {"load_overrides", "runtime_policy", "is_default"} & patch.keys()
            if forbidden:
                raise ValueError(
                    "Router profiles cannot configure load overrides, runtime policy, or default status."
                )
            if "routing_targets" in patch:
                await self._validate_router_targets(profile_id, patch["routing_targets"])
        elif patch.get("routing_targets"):
            raise ValueError("Execution profiles cannot declare routing_targets.")
        for key, value in patch.items():
            if key not in allowed_keys:
                continue
            if key == "category" and value not in ("llm", "tts", "stt", "image", "music"):
                raise ValueError(f"Invalid category '{value}'.")
            if key == "is_default" and value and db_profile.base_model_id is not None:
                await self._clear_default(session, db_profile.base_model_id)
            setattr(db_profile, key, value)

        await session.commit()
        await session.refresh(db_profile)
        self._profiles[profile_id] = db_profile
        logger.info("profile_updated", profile_id=profile_id)
        return db_profile

    async def rename(
        self,
        session: AsyncSession,
        old_profile_id: str,
        new_profile_id: str,
    ) -> ModelProfile:
        """Rename a profile and cascade the change into every router that
        references it in ``routing_targets``.

        The attribution column ``via_router_profile_id`` on ``request_stats``
        is intentionally **not** rewritten: historical rows describe which
        router processed a given request at the time it happened; mutating
        them would falsify history. New requests will naturally use the new
        id.
        """
        if not _is_valid_slug(new_profile_id):
            raise ValueError(
                f"Invalid profile_id '{new_profile_id}': must be lowercase "
                "alphanumeric with dots/dashes/colons."
            )
        if old_profile_id == new_profile_id:
            raise ValueError("new_profile_id matches current id.")
        if new_profile_id in self._profiles:
            raise ValueError(f"Profile '{new_profile_id}' already exists.")

        result = await session.execute(
            select(ModelProfile).where(ModelProfile.profile_id == old_profile_id)
        )
        db_profile = result.scalar_one_or_none()
        if db_profile is None:
            raise ValueError(f"Profile '{old_profile_id}' not found.")

        # Update the row's primary key.
        db_profile.profile_id = new_profile_id

        # Cascade into every other profile whose routing_targets includes the
        # old id. Walk the full registry rather than issuing a jsonb query
        # because we also need the SQLAlchemy session-attached rows.
        other_rows = (
            await session.execute(
                select(ModelProfile).where(ModelProfile.profile_id != new_profile_id)
            )
        ).scalars().all()
        affected_routers: list[str] = []
        for row in other_rows:
            targets = row.routing_targets
            if not isinstance(targets, list) or old_profile_id not in targets:
                continue
            row.routing_targets = [
                new_profile_id if t == old_profile_id else t for t in targets
            ]
            affected_routers.append(row.profile_id)

        await session.commit()
        await session.refresh(db_profile)

        # Rebuild the in-memory cache so every row (including the renamed
        # entries we just mutated) is addressable under its new id.
        self._profiles.pop(old_profile_id, None)
        self._profiles[new_profile_id] = db_profile
        for pid in affected_routers:
            cached = self._profiles.get(pid)
            if cached is not None:
                cached.routing_targets = [
                    new_profile_id if t == old_profile_id else t
                    for t in (cached.routing_targets or [])
                ]
        logger.info(
            "profile_renamed",
            old_profile_id=old_profile_id,
            new_profile_id=new_profile_id,
            affected_routers=affected_routers,
        )
        return db_profile

    async def clone(
        self,
        session: AsyncSession,
        source_profile_id: str,
        new_profile_id: str,
        display_name: str | None = None,
    ) -> ModelProfile:
        """Clone a profile under a new id. The clone keeps every field except
        the primary key; ``is_default`` is forced to False to avoid two
        defaults for the same base model.
        """
        if not _is_valid_slug(new_profile_id):
            raise ValueError(
                f"Invalid profile_id '{new_profile_id}': must be lowercase "
                "alphanumeric with dots/dashes/colons."
            )
        if new_profile_id in self._profiles:
            raise ValueError(f"Profile '{new_profile_id}' already exists.")

        source = self._profiles.get(source_profile_id)
        if source is None:
            raise ValueError(f"Profile '{source_profile_id}' not found.")

        clone = ModelProfile(
            profile_id=new_profile_id,
            profile_kind=source.profile_kind,
            base_model_id=source.base_model_id,
            display_name=display_name if display_name is not None else source.display_name,
            description=source.description,
            category=source.category,
            load_overrides=source.load_overrides,
            request_defaults=source.request_defaults,
            assets=source.assets,
            runtime_policy=source.runtime_policy,
            enabled=source.enabled,
            is_default=False,
            routing_targets=(
                list(source.routing_targets) if source.routing_targets is not None else None
            ),
        )
        session.add(clone)
        await session.flush()
        for preset in source.runtime_presets:
            session.add(
                ProfileRuntimePreset(
                    profile_id=new_profile_id,
                    preset_key=preset.preset_key,
                    label=preset.label,
                    context_mode=preset.context_mode,
                    context_tokens=preset.context_tokens,
                    parallelism=preset.parallelism,
                    kv_cache_dtype=preset.kv_cache_dtype,
                    placement_policy=preset.placement_policy,
                    preferred_gpus=(
                        list(preset.preferred_gpus)
                        if preset.preferred_gpus is not None
                        else None
                    ),
                    execution_policy=preset.execution_policy,
                    load_policy=preset.load_policy,
                    enabled=preset.enabled,
                    is_default=preset.is_default,
                    priority=preset.priority,
                )
            )
        await session.commit()
        clone = await self._reload_profile(session, new_profile_id)
        logger.info(
            "profile_cloned",
            source_profile_id=source_profile_id,
            new_profile_id=new_profile_id,
        )
        return clone

    # ── Runtime presets ───────────────────────────────────────

    async def list_runtime_presets(self, profile_id: str) -> list[ProfileRuntimePreset]:
        """Return a profile's runtime presets in deterministic UI order."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ValueError(f"Profile '{profile_id}' not found.")
        return sorted(
            profile.runtime_presets,
            key=lambda preset: (
                preset.priority,
                preset.context_tokens if preset.context_tokens is not None else 2**63,
                preset.preset_key,
            ),
        )

    async def create_runtime_preset(
        self,
        session: AsyncSession,
        profile_id: str,
        values: dict,
    ) -> ProfileRuntimePreset:
        """Validate and persist a runtime preset for ``profile_id``."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ValueError(f"Profile '{profile_id}' not found.")
        if getattr(profile, "profile_kind", "execution") != "execution":
            raise ValueError("Runtime presets can only be added to execution profiles.")
        validated = RuntimePresetCreate.model_validate(values)
        existing = await self.list_runtime_presets(profile_id)
        if any(p.preset_key == validated.preset_key for p in existing):
            raise ValueError(
                f"Runtime preset '{validated.preset_key}' already exists in profile "
                f"'{profile_id}'."
            )

        payload = validated.model_dump()
        # The first preset becomes the default automatically. This keeps the
        # profile valid while making the create flow pleasant in the UI.
        if not existing:
            payload["is_default"] = True
            payload["enabled"] = True
        elif payload["is_default"]:
            await self._clear_runtime_default(session, profile_id)
            await session.flush()

        preset = ProfileRuntimePreset(profile_id=profile_id, **payload)
        session.add(preset)
        await session.commit()
        await session.refresh(preset)
        await self._reload_profile(session, profile_id)
        logger.info(
            "profile_runtime_preset_created",
            profile_id=profile_id,
            preset_key=preset.preset_key,
        )
        return preset

    async def update_runtime_preset(
        self,
        session: AsyncSession,
        profile_id: str,
        preset_key: str,
        patch: dict,
    ) -> ProfileRuntimePreset:
        """Patch a runtime preset after validating the merged configuration."""
        preset = await self._get_runtime_preset_row(session, profile_id, preset_key)
        current = {
            "preset_key": preset.preset_key,
            "label": preset.label,
            "context_mode": preset.context_mode,
            "context_tokens": preset.context_tokens,
            "parallelism": preset.parallelism,
            "kv_cache_dtype": preset.kv_cache_dtype,
            "placement_policy": preset.placement_policy,
            "preferred_gpus": preset.preferred_gpus,
            "execution_policy": preset.execution_policy,
            "load_policy": preset.load_policy,
            "enabled": preset.enabled,
            "is_default": preset.is_default,
            "priority": preset.priority,
        }
        merged = {**current, **patch}
        validated = RuntimePresetCreate.model_validate(merged)

        if preset.is_default and patch.get("is_default") is False:
            raise ValueError(
                "Set another runtime preset as default instead of clearing the current default."
            )
        if validated.is_default and not preset.is_default:
            await self._clear_runtime_default(session, profile_id)
            await session.flush()

        for key, value in validated.model_dump(exclude={"preset_key"}).items():
            setattr(preset, key, value)
        await session.commit()
        await session.refresh(preset)
        await self._reload_profile(session, profile_id)
        logger.info(
            "profile_runtime_preset_updated",
            profile_id=profile_id,
            preset_key=preset_key,
        )
        return preset

    async def delete_runtime_preset(
        self,
        session: AsyncSession,
        profile_id: str,
        preset_key: str,
    ) -> None:
        """Delete a non-default preset, or the only preset in a profile."""
        preset = await self._get_runtime_preset_row(session, profile_id, preset_key)
        count = len(await self.list_runtime_presets(profile_id))
        if preset.is_default and count > 1:
            raise ValueError(
                "Set another runtime preset as default before deleting the current default."
            )
        await session.delete(preset)
        await session.commit()
        await self._reload_profile(session, profile_id)
        logger.info(
            "profile_runtime_preset_deleted",
            profile_id=profile_id,
            preset_key=preset_key,
        )

    async def delete(self, session: AsyncSession, profile_id: str) -> None:
        """Delete a profile and remove from cache."""
        referenced_by = [
            profile.profile_id
            for profile in self._profiles.values()
            if profile.profile_id != profile_id
            and isinstance(profile.routing_targets, list)
            and profile_id in profile.routing_targets
        ]
        if referenced_by:
            raise ValueError(
                f"Profile '{profile_id}' is referenced by router(s): "
                + ", ".join(sorted(referenced_by))
            )
        result = await session.execute(
            select(ModelProfile).where(ModelProfile.profile_id == profile_id)
        )
        db_profile = result.scalar_one_or_none()
        if db_profile is None:
            raise ValueError(f"Profile '{profile_id}' not found.")

        await session.delete(db_profile)
        await session.commit()
        self._profiles.pop(profile_id, None)
        logger.info("profile_deleted", profile_id=profile_id)

    async def set_default(
        self, session: AsyncSession, base_model_id: str, profile_id: str
    ) -> ModelProfile:
        """Set a profile as default for its base model, clearing any previous default."""
        result = await session.execute(
            select(ModelProfile).where(ModelProfile.profile_id == profile_id)
        )
        db_profile = result.scalar_one_or_none()
        if db_profile is None:
            raise ValueError(f"Profile '{profile_id}' not found.")
        if db_profile.base_model_id != base_model_id:
            raise ValueError(f"Profile '{profile_id}' does not belong to model '{base_model_id}'.")

        await self._clear_default(session, base_model_id)
        db_profile.is_default = True
        await session.commit()
        await session.refresh(db_profile)
        # Refresh all profiles for this model in cache
        await self._refresh_model_profiles(session, base_model_id)
        return db_profile

    async def _validate_router_targets(
        self,
        profile_id: str,
        routing_targets: list[str] | None,
    ) -> None:
        """Validate a router's ordered destinations against the registry."""

        if not isinstance(routing_targets, list) or not routing_targets:
            raise ValueError("Router profiles require at least one routing target.")
        if len(set(routing_targets)) != len(routing_targets):
            raise ValueError("Router routing_targets must not contain duplicates.")
        if profile_id in routing_targets:
            raise ValueError("A router cannot target itself.")
        missing = [target for target in routing_targets if target not in self._profiles]
        if missing:
            raise ValueError(
                "Router target profile(s) not found: " + ", ".join(missing)
            )

    async def update_assets(
        self,
        session: AsyncSession,
        profile_id: str,
        assets: dict | None,
    ) -> ModelProfile:
        """Update the assets JSONB of a profile."""
        result = await session.execute(
            select(ModelProfile).where(ModelProfile.profile_id == profile_id)
        )
        db_profile = result.scalar_one_or_none()
        if db_profile is None:
            raise ValueError(f"Profile '{profile_id}' not found.")
        db_profile.assets = assets
        await session.commit()
        await session.refresh(db_profile)
        self._profiles[profile_id] = db_profile
        return db_profile

    # ── Auto-seed diarized profiles for whisper models ────────

    async def ensure_diarized_profiles(self, session: AsyncSession) -> int:
        """For each base whisper model without a diarized profile, create one.

        Returns the number of profiles created.
        """
        import re

        from ocabra.core.model_manager_helpers import build_diarized_extra_config

        result = await session.execute(
            select(ModelConfig).where(ModelConfig.backend_type == "whisper")
        )
        whisper_models = [m for m in result.scalars().all() if "::" not in m.model_id]

        created = 0
        for model in whisper_models:
            # Check if a diarized profile already exists for this base model
            existing = [
                p
                for p in self._profiles.values()
                if p.base_model_id == model.model_id
                and p.load_overrides
                and p.load_overrides.get("diarization_enabled") is True
            ]
            if existing:
                continue

            # Derive slug from model display name
            raw_name = model.display_name or model.model_id
            if "/" in model.model_id:
                parts = model.model_id.split("/", 1)
                raw_name = model.display_name or parts[1] if len(parts) > 1 else raw_name
            slug = re.sub(r"[^a-z0-9\-.]", "", raw_name.lower().replace("/", "-").replace(" ", "-"))
            slug = re.sub(r"-{2,}", "-", slug).strip("-")
            profile_id = f"{slug}-diarized"

            if profile_id in self._profiles:
                continue

            try:
                await self.create(
                    session,
                    profile_id=profile_id,
                    base_model_id=model.model_id,
                    display_name=f"{model.display_name or raw_name} (Diarized)",
                    category="stt",
                    load_overrides=build_diarized_extra_config(model.extra_config),
                    request_defaults={"diarize": True},
                    enabled=True,
                    is_default=False,
                )
                created += 1
                logger.info(
                    "diarized_profile_auto_created",
                    profile_id=profile_id,
                    base_model_id=model.model_id,
                )
            except Exception as exc:
                logger.warning(
                    "diarized_profile_auto_create_failed",
                    profile_id=profile_id,
                    base_model_id=model.model_id,
                    error=str(exc),
                )
        return created

    # ── Auto-seed default profiles for models without any ─────

    _BACKEND_CATEGORY: dict[str, str] = {
        "vllm": "llm", "llama_cpp": "llm", "sglang": "llm",
        "tensorrt_llm": "llm", "bitnet": "llm", "ollama": "llm",
        "whisper": "stt", "vibeasr": "stt", "tts": "tts", "voxtral": "tts",
        "chatterbox": "tts", "piper": "tts", "diffusers": "image", "acestep": "music",
    }

    async def ensure_default_profiles(self, session: AsyncSession) -> int:
        """Create a default profile for every model_config that has no profiles.

        Returns the number of profiles created.
        """
        result = await session.execute(select(ModelConfig))
        all_models = result.scalars().all()

        created = 0
        for model in all_models:
            if await self._ensure_default_profile_for(session, model) is not None:
                created += 1
        return created

    async def _ensure_default_profile_for(
        self, session: AsyncSession, model: ModelConfig
    ) -> str | None:
        """Create a default profile for ``model`` if it has none.

        Returns the created profile_id (or ``None`` when a profile already
        exists, the model kind is skipped, or slug derivation fails). Errors
        during creation are logged and swallowed — the model is still
        registered, it just won't be visible under a friendly profile_id
        until the next reconciliation.
        """
        import re

        # A model already has a profile? Nothing to do.
        if any(p.base_model_id == model.model_id for p in self._profiles.values()):
            return None
        # Skip legacy diarized variants (but not GGUF models that use :: as file separator).
        if "::" in model.model_id and model.backend_type not in ("bitnet", "llama_cpp"):
            return None

        raw_name = model.display_name or model.model_id
        if "/" in model.model_id:
            parts = model.model_id.split("/", 1)
            if parts[0] in self._BACKEND_CATEGORY:
                raw_name = model.display_name or parts[1]
        slug = re.sub(
            r"[^a-z0-9\-.:]",
            "",
            raw_name.lower().replace("/", "-").replace("_", "-").replace(" ", "-"),
        )
        slug = re.sub(r"-{2,}", "-", slug).strip("-")
        if not slug:
            slug = re.sub(r"[^a-z0-9\-.:]", "", model.model_id.lower().replace("/", "-"))
            slug = re.sub(r"-{2,}", "-", slug).strip("-")
        if not slug:
            return None

        if slug in self._profiles or slug in self._aliases:
            # Display names are not unique ("Flux", "Whisper", …).  Skipping
            # here silently leaves the newly registered model invisible to the
            # public API, so derive a deterministic collision-safe id instead.
            suffix = hashlib.sha256(model.model_id.encode()).hexdigest()[:8]
            stem = slug[: _MAX_PROFILE_ID_LEN - len(suffix) - 1].rstrip("-.: ")
            slug = f"{stem or 'model'}-{suffix}"
            if slug in self._profiles or slug in self._aliases:
                return None

        category = self._BACKEND_CATEGORY.get(model.backend_type or "", "llm")
        try:
            await self.create(
                session,
                profile_id=slug,
                base_model_id=model.model_id,
                display_name=model.display_name or raw_name,
                category=category,
                enabled=True,
                is_default=True,
            )
            logger.info(
                "default_profile_auto_created",
                profile_id=slug,
                base_model_id=model.model_id,
            )
            return slug
        except Exception as exc:
            logger.warning(
                "default_profile_auto_create_failed",
                profile_id=slug,
                base_model_id=model.model_id,
                error=str(exc),
            )
            return None

    async def on_model_added(self, model_id: str) -> str | None:
        """Post-registration hook: create a default profile for ``model_id``.

        Meant to be called from every ``add_model`` call site (POST /models,
        the download completion path, TRT-LLM engine registration, …).
        Previously ``ensure_default_profiles`` only ran on startup and after
        the Ollama inventory loop, so models added at runtime by other paths
        stayed invisible in /v1/models until the next restart. Opens its own
        DB session so callers don't need to thread one through. Failures are
        logged and swallowed to keep the registration path atomic.
        """
        from ocabra.database import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(ModelConfig).where(ModelConfig.model_id == model_id)
                )
                model = result.scalar_one_or_none()
                if model is None:
                    return None
                created = await self._ensure_default_profile_for(session, model)
                await session.commit()
                return created
        except Exception as exc:
            logger.warning(
                "on_model_added_failed",
                model_id=model_id,
                error=str(exc),
            )
            return None

    # ── Helpers ───────────────────────────────────────────────

    async def _clear_default(self, session: AsyncSession, base_model_id: str) -> None:
        """Clear is_default for all profiles of a given base model."""
        result = await session.execute(
            select(ModelProfile).where(
                ModelProfile.base_model_id == base_model_id,
                ModelProfile.is_default.is_(True),
            )
        )
        for p in result.scalars().all():
            p.is_default = False

    async def _clear_runtime_default(self, session: AsyncSession, profile_id: str) -> None:
        """Clear the current default runtime preset inside one profile."""
        result = await session.execute(
            select(ProfileRuntimePreset).where(
                ProfileRuntimePreset.profile_id == profile_id,
                ProfileRuntimePreset.is_default.is_(True),
            )
        )
        for preset in result.scalars().all():
            preset.is_default = False

    async def _get_runtime_preset_row(
        self,
        session: AsyncSession,
        profile_id: str,
        preset_key: str,
    ) -> ProfileRuntimePreset:
        if profile_id not in self._profiles:
            raise ValueError(f"Profile '{profile_id}' not found.")
        result = await session.execute(
            select(ProfileRuntimePreset).where(
                ProfileRuntimePreset.profile_id == profile_id,
                ProfileRuntimePreset.preset_key == preset_key,
            )
        )
        preset = result.scalar_one_or_none()
        if preset is None:
            raise ValueError(
                f"Runtime preset '{preset_key}' not found in profile '{profile_id}'."
            )
        return preset

    async def _reload_profile(
        self,
        session: AsyncSession,
        profile_id: str,
    ) -> ModelProfile:
        result = await session.execute(
            select(ModelProfile)
            .options(selectinload(ModelProfile.runtime_presets))
            .where(ModelProfile.profile_id == profile_id)
        )
        profile = result.scalar_one_or_none()
        if profile is None:
            raise ValueError(f"Profile '{profile_id}' not found.")
        self._profiles[profile_id] = profile
        return profile

    async def _refresh_model_profiles(self, session: AsyncSession, base_model_id: str) -> None:
        """Refresh cache for all profiles of a base model."""
        result = await session.execute(
            select(ModelProfile)
            .options(selectinload(ModelProfile.runtime_presets))
            .where(ModelProfile.base_model_id == base_model_id)
        )
        for p in result.scalars().all():
            self._profiles[p.profile_id] = p
