"""Tests for config update whitelist — ServerConfigPatch validation.

Ensures that:
- Allowed mutable fields are accepted and applied
- Read-only fields (models_dir) raise HTTPException
- Unknown/extra fields are rejected by Pydantic (extra="forbid")
- Sensitive fields like database_url are not in the schema at all
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from ocabra.api.internal import config as config_api
from ocabra.api.internal.config import ServerConfigPatch
from ocabra.config import settings


class TestServerConfigPatchSchema:
    """Pydantic schema-level validation."""

    def test_accepts_valid_mutable_fields(self):
        """Known mutable fields are accepted."""
        patch = ServerConfigPatch.model_validate({
            "idleTimeoutSeconds": 300,
            "inferenceRequestTimeoutSeconds": 900,
            "logLevel": "DEBUG",
            "vllmGpuMemoryUtilization": 0.85,
            "defaultGpuIndex": 0,
        })
        assert patch.idle_timeout_seconds == 300
        assert patch.inference_request_timeout_seconds == 900
        assert patch.log_level == "DEBUG"
        assert patch.vllm_gpu_memory_utilization == 0.85

    def test_rejects_unknown_fields(self):
        """Extra fields not in schema are rejected (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ServerConfigPatch.model_validate({"database_url": "postgres://evil"})
        assert "database_url" in str(exc_info.value)

    def test_rejects_arbitrary_extra_field(self):
        with pytest.raises(ValidationError):
            ServerConfigPatch.model_validate({"some_random_key": "value"})

    def test_rejects_secret_key_field(self):
        """Fields like secret_key are not part of the config patch schema."""
        with pytest.raises(ValidationError):
            ServerConfigPatch.model_validate({"secret_key": "hunter2"})

    def test_rejects_redis_url(self):
        with pytest.raises(ValidationError):
            ServerConfigPatch.model_validate({"redis_url": "redis://evil:6379"})

    def test_empty_patch_is_valid(self):
        """An empty patch should be valid (no fields updated)."""
        patch = ServerConfigPatch.model_validate({})
        dumped = patch.model_dump(exclude_unset=True)
        assert dumped == {}

    def test_models_dir_is_accepted_by_schema(self):
        """models_dir IS in schema but rejected at runtime by patch_config()."""
        patch = ServerConfigPatch.model_validate({"modelsDir": "/tmp/models"})
        assert patch.models_dir == "/tmp/models"


class TestPatchConfigRuntime:
    """Runtime behavior of the patch_config endpoint."""

    @pytest.mark.asyncio
    async def test_models_dir_rejected_at_runtime(self):
        """Attempting to change modelsDir raises 400."""
        from fastapi import HTTPException

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        patch = ServerConfigPatch.model_validate({"modelsDir": "/tmp/evil"})

        with pytest.raises(HTTPException) as exc_info:
            await config_api.patch_config(patch, request)

        assert exc_info.value.status_code == 400
        assert "cannot be changed at runtime" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_idle_timeout_applied(self):
        """idle_timeout_seconds is applied to settings."""
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        previous = settings.idle_timeout_seconds

        try:
            patch = ServerConfigPatch.model_validate({"idleTimeoutSeconds": 999})

            with patch_db_and_schedules():
                result = await config_api.patch_config(patch, request)

            assert result["idleTimeoutSeconds"] == 999
            assert settings.idle_timeout_seconds == 999
        finally:
            settings.idle_timeout_seconds = previous

    @pytest.mark.asyncio
    async def test_log_level_applied(self):
        """log_level change is applied."""
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        previous = settings.log_level

        try:
            patch = ServerConfigPatch.model_validate({"logLevel": "WARNING"})

            with patch_db_and_schedules():
                result = await config_api.patch_config(patch, request)

            assert result["logLevel"] == "WARNING"
            assert settings.log_level == "WARNING"
        finally:
            settings.log_level = previous

    @pytest.mark.asyncio
    async def test_inference_timeout_applied(self):
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        previous = settings.inference_request_timeout_seconds

        try:
            patch = ServerConfigPatch.model_validate({
                "inferenceRequestTimeoutSeconds": 1200,
            })

            with patch_db_and_schedules():
                result = await config_api.patch_config(patch, request)

            assert result["inferenceRequestTimeoutSeconds"] == 1200
            assert settings.inference_request_timeout_seconds == 1200
        finally:
            settings.inference_request_timeout_seconds = previous

    @pytest.mark.asyncio
    async def test_download_dir_stored_in_overrides(self):
        """download_dir is stored in request-level overrides, not settings."""
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        patch = ServerConfigPatch.model_validate({"downloadDir": "/tmp/dl"})

        with patch_db_and_schedules():
            result = await config_api.patch_config(patch, request)

        assert result["downloadDir"] == "/tmp/dl"

    @pytest.mark.asyncio
    async def test_default_reasoning_effort_applied(self):
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        previous = settings.default_reasoning_effort

        try:
            patch = ServerConfigPatch.model_validate({"defaultReasoningEffort": "high"})

            with patch_db_and_schedules():
                result = await config_api.patch_config(patch, request)

            assert result["defaultReasoningEffort"] == "high"
            assert settings.default_reasoning_effort == "high"
        finally:
            settings.default_reasoning_effort = previous

    @pytest.mark.asyncio
    async def test_default_reasoning_effort_rejects_invalid_value(self):
        from fastapi import HTTPException

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        patch = ServerConfigPatch.model_validate({"defaultReasoningEffort": "extreme"})

        with patch_db_and_schedules(), pytest.raises(HTTPException) as exc_info:
            await config_api.patch_config(patch, request)

        assert exc_info.value.status_code == 400


class TestRetiredRouterFallbackDelay:
    """BUG-32: ``router_fallback_delay_ms`` was editable and nothing read it.
    It is gone from the settings and the GET response; PATCH still accepts it
    (ignored, not persisted) so old clients don't get a 422, and a value
    already stored in server_config doesn't break startup."""

    def test_not_a_setting_anymore(self):
        assert not hasattr(settings, "router_fallback_delay_ms")

    @pytest.mark.asyncio
    async def test_patch_accepts_and_ignores_it(self):
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        patch_body = ServerConfigPatch.model_validate({"routerFallbackDelayMs": 500})
        save = AsyncMock()
        with (
            patch.object(config_api, "save_override", new=save),
            patch.object(config_api, "AsyncSessionLocal", new=_FakeSessionLocal),
            patch.object(config_api, "_load_global_schedules", new=AsyncMock(return_value=[])),
        ):
            result = await config_api.patch_config(patch_body, request)
        assert "routerFallbackDelayMs" not in result
        save.assert_not_awaited()

    def test_stored_override_is_skipped_at_startup(self):
        from ocabra.config import Settings
        from ocabra.db.server_config import apply_overrides_to_settings

        fresh = Settings()
        apply_overrides_to_settings(
            fresh, {"router_fallback_delay_ms": "200", "idle_timeout_seconds": "123"}
        )
        assert fresh.idle_timeout_seconds == 123
        assert not hasattr(fresh, "router_fallback_delay_ms")


class TestRetiredLiteLLM:
    """LiteLLM support was removed. Its settings are gone from the settings
    and the GET response; PATCH still accepts the old keys (ignored, not
    persisted) so old clients don't get a 422, and values already stored in
    server_config don't break startup."""

    _KEYS = ("litellm_base_url", "litellm_admin_key", "litellm_auto_sync")

    def test_not_settings_anymore(self):
        for key in self._KEYS:
            assert not hasattr(settings, key)

    @pytest.mark.asyncio
    async def test_patch_accepts_and_ignores_them(self):
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        patch_body = ServerConfigPatch.model_validate(
            {
                "litellmBaseUrl": "http://litellm:4000",
                "litellmAdminKey": "sk-old",
                "litellmAutoSync": True,
            }
        )
        save = AsyncMock()
        with (
            patch.object(config_api, "save_override", new=save),
            patch.object(config_api, "AsyncSessionLocal", new=_FakeSessionLocal),
            patch.object(config_api, "_load_global_schedules", new=AsyncMock(return_value=[])),
        ):
            result = await config_api.patch_config(patch_body, request)
        for key in ("litellmBaseUrl", "litellmAdminKey", "litellmAutoSync"):
            assert key not in result
        save.assert_not_awaited()

    def test_stored_overrides_are_skipped_at_startup(self):
        from ocabra.config import Settings
        from ocabra.db.server_config import apply_overrides_to_settings

        fresh = Settings()
        apply_overrides_to_settings(
            fresh,
            {
                "litellm_base_url": '"http://litellm:4000"',
                "litellm_admin_key": '"sk-old"',
                "litellm_auto_sync": "true",
                "idle_timeout_seconds": "123",
            },
        )
        assert fresh.idle_timeout_seconds == 123
        for key in self._KEYS:
            assert not hasattr(fresh, key)


def test_langfuse_settings_retired_and_stored_overrides_skipped():
    """Langfuse support was removed: no ``langfuse_*`` settings remain, and a
    value left in server_config (or LANGFUSE_* in .env) doesn't break startup."""
    from ocabra.config import Settings
    from ocabra.db.server_config import apply_overrides_to_settings

    fresh = Settings()
    apply_overrides_to_settings(
        fresh, {"langfuse_enabled": "true", "langfuse_host": '"https://x"'}
    )
    assert not [name for name in Settings.model_fields if name.startswith("langfuse_")]
    assert not hasattr(fresh, "langfuse_enabled")


def patch_db_and_schedules():
    """Context manager to mock database persistence and schedule loading."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with (
            patch.object(config_api, "save_override", new=AsyncMock()),
            patch.object(config_api, "AsyncSessionLocal", new=_FakeSessionLocal),
            patch.object(config_api, "_load_global_schedules", new=AsyncMock(return_value=[])),
        ):
            yield

    return _ctx()


class _FakeSessionLocal:
    """Minimal async context manager that mimics AsyncSessionLocal."""

    def __init__(self):
        pass

    async def __aenter__(self):
        return AsyncMock()

    async def __aexit__(self, *args):
        pass
