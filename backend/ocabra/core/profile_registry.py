"""In-memory registry for model profiles with DB-backed persistence."""

from __future__ import annotations

import re

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ocabra.db.model_config import ModelConfig, ModelProfile

logger = structlog.get_logger(__name__)

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")
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

    # ── Bulk load ────────────────────────────────────────────

    async def load_all(self, session: AsyncSession) -> None:
        """Load all profiles from the database into memory."""
        result = await session.execute(select(ModelProfile))
        rows = result.scalars().all()
        self._profiles = {p.profile_id: p for p in rows}
        logger.info("profile_registry_loaded", count=len(self._profiles))

    # ── Read operations (from cache) ─────────────────────────

    async def get(self, profile_id: str) -> ModelProfile | None:
        """Return a single profile by id, or None."""
        return self._profiles.get(profile_id)

    async def list_by_model(self, base_model_id: str) -> list[ModelProfile]:
        """Return all profiles for a given base model."""
        return [p for p in self._profiles.values() if p.base_model_id == base_model_id]

    async def list_enabled(self) -> list[ModelProfile]:
        """Return all enabled profiles."""
        return [p for p in self._profiles.values() if p.enabled]

    async def list_all(self) -> list[ModelProfile]:
        """Return all profiles."""
        return list(self._profiles.values())

    # ── Write operations (write-through) ─────────────────────

    async def create(
        self,
        session: AsyncSession,
        *,
        profile_id: str,
        base_model_id: str,
        display_name: str | None = None,
        description: str | None = None,
        category: str = "llm",
        load_overrides: dict | None = None,
        request_defaults: dict | None = None,
        assets: dict | None = None,
        enabled: bool = True,
        is_default: bool = False,
    ) -> ModelProfile:
        """Create a new profile, validate constraints, persist, and update cache."""
        if not _is_valid_slug(profile_id):
            raise ValueError(
                f"Invalid profile_id '{profile_id}': must be lowercase alphanumeric "
                "with hyphens/dots, starting with an alphanumeric character."
            )
        if profile_id in self._profiles:
            raise ValueError(f"Profile '{profile_id}' already exists.")

        if category not in ("llm", "tts", "stt", "image", "music"):
            raise ValueError(f"Invalid category '{category}'.")

        # Verify base model exists
        result = await session.execute(
            select(ModelConfig).where(ModelConfig.model_id == base_model_id)
        )
        if result.scalar_one_or_none() is None:
            raise ValueError(f"Base model '{base_model_id}' not found.")

        # If marking as default, clear existing default for this base model
        if is_default:
            await self._clear_default(session, base_model_id)

        profile = ModelProfile(
            profile_id=profile_id,
            base_model_id=base_model_id,
            display_name=display_name,
            description=description,
            category=category,
            load_overrides=load_overrides,
            request_defaults=request_defaults,
            assets=assets,
            enabled=enabled,
            is_default=is_default,
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
        }
        for key, value in patch.items():
            if key not in allowed_keys:
                continue
            if key == "category" and value not in ("llm", "tts", "stt", "image", "music"):
                raise ValueError(f"Invalid category '{value}'.")
            if key == "is_default" and value:
                await self._clear_default(session, db_profile.base_model_id)
            setattr(db_profile, key, value)

        await session.commit()
        await session.refresh(db_profile)
        self._profiles[profile_id] = db_profile
        logger.info("profile_updated", profile_id=profile_id)
        return db_profile

    async def delete(self, session: AsyncSession, profile_id: str) -> None:
        """Delete a profile and remove from cache."""
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
        "whisper": "stt", "tts": "tts", "voxtral": "tts",
        "chatterbox": "tts", "diffusers": "image", "acestep": "music",
    }

    async def ensure_default_profiles(self, session: AsyncSession) -> int:
        """Create a default profile for every model_config that has no profiles.

        Returns the number of profiles created.
        """
        import re

        result = await session.execute(select(ModelConfig))
        all_models = result.scalars().all()

        # Models that already have at least one profile
        models_with_profiles = {p.base_model_id for p in self._profiles.values()}

        created = 0
        for model in all_models:
            if model.model_id in models_with_profiles:
                continue
            if "::" in model.model_id:
                continue  # skip legacy diarized variants

            # Derive slug
            raw_name = model.display_name or model.model_id
            if "/" in model.model_id:
                parts = model.model_id.split("/", 1)
                if parts[0] in self._BACKEND_CATEGORY:
                    raw_name = model.display_name or parts[1]
            slug = re.sub(r"[^a-z0-9\-.]", "", raw_name.lower().replace("/", "-").replace("_", "-").replace(" ", "-"))
            slug = re.sub(r"-{2,}", "-", slug).strip("-")
            if not slug:
                slug = re.sub(r"[^a-z0-9\-.]", "", model.model_id.lower().replace("/", "-"))
                slug = re.sub(r"-{2,}", "-", slug).strip("-")
            if not slug or slug in self._profiles:
                continue

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
                created += 1
                logger.info("default_profile_auto_created", profile_id=slug, base_model_id=model.model_id)
            except Exception as exc:
                logger.warning(
                    "default_profile_auto_create_failed",
                    profile_id=slug,
                    base_model_id=model.model_id,
                    error=str(exc),
                )
        return created

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

    async def _refresh_model_profiles(self, session: AsyncSession, base_model_id: str) -> None:
        """Refresh cache for all profiles of a base model."""
        result = await session.execute(
            select(ModelProfile).where(ModelProfile.base_model_id == base_model_id)
        )
        for p in result.scalars().all():
            self._profiles[p.profile_id] = p
