"""Tests for Model Profiles: CRUD, cascade, uniqueness, assets, is_default constraint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ocabra.core.profile_registry import ProfileRegistry, _is_valid_slug

# ── Slug validation ──────────────────────────────────────────


class TestSlugValidation:
    def test_valid_slugs(self):
        assert _is_valid_slug("my-profile")
        assert _is_valid_slug("qwen3-8b-chat")
        assert _is_valid_slug("whisper-large-v3")
        assert _is_valid_slug("a")
        assert _is_valid_slug("model.v2")

    def test_invalid_slugs(self):
        assert not _is_valid_slug("")
        assert not _is_valid_slug("-starts-with-dash")
        assert not _is_valid_slug("Has/Slash")
        assert not _is_valid_slug("HAS_UPPER")
        assert not _is_valid_slug("a" * 513)
        assert not _is_valid_slug(".dotstart")


# ── Helpers for in-memory testing ────────────────────────────


def _make_profile(**overrides):
    """Create a SimpleNamespace that looks enough like a ModelProfile for cache tests."""
    defaults = {
        "profile_id": "test-profile",
        "base_model_id": "vllm/my-model",
        "display_name": "Test Profile",
        "description": None,
        "category": "llm",
        "load_overrides": None,
        "request_defaults": None,
        "assets": None,
        "enabled": True,
        "is_default": False,
        "created_at": None,
        "updated_at": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestProfileRegistryCache:
    """Test the in-memory cache behaviour of ProfileRegistry (no DB)."""

    @pytest.fixture
    def registry(self):
        r = ProfileRegistry()
        return r

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, registry):
        result = await registry.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_model_filters(self, registry):
        p1 = _make_profile(profile_id="a", base_model_id="vllm/m1")
        p2 = _make_profile(profile_id="b", base_model_id="vllm/m2")
        p3 = _make_profile(profile_id="c", base_model_id="vllm/m1")
        registry._profiles = {"a": p1, "b": p2, "c": p3}

        result = await registry.list_by_model("vllm/m1")
        assert len(result) == 2
        assert {r.profile_id for r in result} == {"a", "c"}

    @pytest.mark.asyncio
    async def test_list_enabled_filters(self, registry):
        p1 = _make_profile(profile_id="a", enabled=True)
        p2 = _make_profile(profile_id="b", enabled=False)
        registry._profiles = {"a": p1, "b": p2}

        result = await registry.list_enabled()
        assert len(result) == 1
        assert result[0].profile_id == "a"

    @pytest.mark.asyncio
    async def test_list_all(self, registry):
        p1 = _make_profile(profile_id="a")
        p2 = _make_profile(profile_id="b")
        registry._profiles = {"a": p1, "b": p2}

        result = await registry.list_all()
        assert len(result) == 2


# ── DB-backed tests using mocked session ─────────────────────


class _FakeScalarResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items

    def scalars(self):
        return self


class _FakeResult:
    def __init__(self, item=None, items=None):
        self._item = item
        self._items = items or []

    def scalar_one_or_none(self):
        return self._item

    def scalars(self):
        return _FakeScalarResult(self._items)


def _make_session(*, model_exists=True, profiles=None):
    """Create a mock AsyncSession."""
    session = AsyncMock()
    profiles = profiles or []

    # For model existence check
    model_result = _FakeResult(
        item=SimpleNamespace(model_id="vllm/test-model") if model_exists else None
    )
    # For profile lookups
    profile_result = _FakeResult(items=profiles)

    # execute returns different results depending on the query
    call_count = {"n": 0}

    async def _execute(stmt):
        call_count["n"] += 1
        # First call is usually the model existence check for create
        if call_count["n"] == 1:
            return model_result
        return profile_result

    session.execute = _execute

    async def _commit():
        pass

    session.commit = _commit

    async def _refresh(obj):
        pass

    session.refresh = _refresh

    async def _delete(obj):
        pass

    session.delete = _delete

    def _add(obj):
        pass

    session.add = _add

    return session


class TestProfileRegistryCRUD:
    @pytest.mark.asyncio
    async def test_create_validates_slug(self):
        registry = ProfileRegistry()
        session = _make_session()
        with pytest.raises(ValueError, match="Invalid profile_id"):
            await registry.create(
                session,
                profile_id="INVALID/slug",
                base_model_id="vllm/test-model",
            )

    @pytest.mark.asyncio
    async def test_create_rejects_duplicate(self):
        registry = ProfileRegistry()
        registry._profiles["existing"] = _make_profile(profile_id="existing")
        session = _make_session()
        with pytest.raises(ValueError, match="already exists"):
            await registry.create(
                session,
                profile_id="existing",
                base_model_id="vllm/test-model",
            )

    @pytest.mark.asyncio
    async def test_create_rejects_invalid_category(self):
        registry = ProfileRegistry()
        session = _make_session()
        with pytest.raises(ValueError, match="Invalid category"):
            await registry.create(
                session,
                profile_id="valid-slug",
                base_model_id="vllm/test-model",
                category="invalid",
            )

    @pytest.mark.asyncio
    async def test_create_rejects_missing_base_model(self):
        registry = ProfileRegistry()
        session = _make_session(model_exists=False)
        with pytest.raises(ValueError, match="not found"):
            await registry.create(
                session,
                profile_id="valid-slug",
                base_model_id="vllm/nonexistent",
            )

    @pytest.mark.asyncio
    async def test_create_success(self):
        registry = ProfileRegistry()
        session = _make_session(model_exists=True)
        profile = await registry.create(
            session,
            profile_id="my-new-profile",
            base_model_id="vllm/test-model",
            display_name="My New Profile",
            category="llm",
        )
        assert profile.profile_id == "my-new-profile"
        assert profile.base_model_id == "vllm/test-model"
        assert "my-new-profile" in registry._profiles

    @pytest.mark.asyncio
    async def test_delete_removes_from_cache(self):
        registry = ProfileRegistry()
        fake_profile = _make_profile(profile_id="to-delete")
        registry._profiles["to-delete"] = fake_profile

        session = AsyncMock()

        async def _execute(stmt):
            return _FakeResult(item=fake_profile)

        session.execute = _execute

        async def _commit():
            pass

        session.commit = _commit

        async def _delete(obj):
            pass

        session.delete = _delete

        await registry.delete(session, "to-delete")
        assert "to-delete" not in registry._profiles

    @pytest.mark.asyncio
    async def test_delete_nonexistent_raises(self):
        registry = ProfileRegistry()
        session = AsyncMock()

        async def _execute(stmt):
            return _FakeResult(item=None)

        session.execute = _execute
        with pytest.raises(ValueError, match="not found"):
            await registry.delete(session, "nonexistent")


# ── Asset path traversal tests ───────────────────────────────


class TestAssetPathTraversal:
    def test_path_within_base(self):
        from ocabra.api.internal.profiles import _is_path_within_base

        base = Path("/data/profiles")
        assert _is_path_within_base(Path("/data/profiles/my-profile/audio.wav"), base)
        assert not _is_path_within_base(Path("/data/profiles/../etc/passwd"), base)
        assert not _is_path_within_base(Path("/etc/passwd"), base)

    def test_allowed_extensions(self):
        from ocabra.api.internal.profiles import ALLOWED_ASSET_EXTENSIONS

        assert ".wav" in ALLOWED_ASSET_EXTENSIONS
        assert ".mp3" in ALLOWED_ASSET_EXTENSIONS
        assert ".safetensors" in ALLOWED_ASSET_EXTENSIONS
        assert ".exe" not in ALLOWED_ASSET_EXTENSIONS
        assert ".sh" not in ALLOWED_ASSET_EXTENSIONS


# ── is_default constraint tests ──────────────────────────────


class TestIsDefaultConstraint:
    @pytest.mark.asyncio
    async def test_create_with_is_default_clears_previous(self):
        """When creating a profile with is_default=True, existing defaults are cleared."""
        registry = ProfileRegistry()

        # Pre-populate with an existing default
        existing_default = _make_profile(
            profile_id="old-default",
            base_model_id="vllm/test-model",
            is_default=True,
        )
        registry._profiles["old-default"] = existing_default

        # Mock session — create() first checks model exists, then calls _clear_default
        session = AsyncMock()
        call_count = {"n": 0}

        async def _execute(stmt):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: model existence check (select ModelConfig)
                return _FakeResult(item=SimpleNamespace(model_id="vllm/test-model"))
            if call_count["n"] == 2:
                # Second call: _clear_default query (select existing defaults)
                return _FakeResult(items=[existing_default])
            # Subsequent: empty
            return _FakeResult(items=[])

        session.execute = _execute

        async def _commit():
            pass

        session.commit = _commit

        async def _refresh(obj):
            pass

        session.refresh = _refresh
        session.add = MagicMock()

        profile = await registry.create(
            session,
            profile_id="new-default",
            base_model_id="vllm/test-model",
            is_default=True,
        )
        assert profile.is_default is True
        assert "new-default" in registry._profiles


# ── Migration slug helper test ───────────────────────────────


class TestMigrationSlugify:
    def test_slugify(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "migration_0010",
            str(Path(__file__).resolve().parent.parent / "alembic" / "versions" / "0010_model_profiles.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert mod._slugify("Qwen/Qwen3-8B") == "qwen-qwen3-8b"
        assert (
            mod._slugify("meta-llama/Llama-3.3-70B-Instruct") == "meta-llama-llama-3.3-70b-instruct"
        )
        assert mod._slugify("whisper-large-v3") == "whisper-large-v3"
        assert mod._slugify("My Model Name") == "my-model-name"
        assert mod._slugify("a/b/c") == "a-b-c"

    def test_backend_category_mapping(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "migration_0010",
            str(Path(__file__).resolve().parent.parent / "alembic" / "versions" / "0010_model_profiles.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert mod._BACKEND_CATEGORY["vllm"] == "llm"
        assert mod._BACKEND_CATEGORY["whisper"] == "stt"
        assert mod._BACKEND_CATEGORY["tts"] == "tts"
        assert mod._BACKEND_CATEGORY["diffusers"] == "image"
        assert mod._BACKEND_CATEGORY["acestep"] == "music"


# ── ProfileOut schema test ───────────────────────────────────


class TestProfileSchema:
    def test_profile_out_from_attributes(self):
        from ocabra.schemas.profiles import ProfileOut

        profile = _make_profile(
            profile_id="test",
            base_model_id="vllm/model",
            category="llm",
            enabled=True,
            is_default=False,
        )
        out = ProfileOut.model_validate(profile)
        assert out.profile_id == "test"
        assert out.base_model_id == "vllm/model"

    def test_profile_create_rejects_extra(self):
        from pydantic import ValidationError

        from ocabra.schemas.profiles import ProfileCreate

        with pytest.raises(ValidationError):
            ProfileCreate(profile_id="x", extra_field="bad")
