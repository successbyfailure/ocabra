"""Tests for profile resolution, request_defaults merge, assets injection,
and worker key logic.

These tests use lightweight mocks — no database, no real workers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Lightweight stubs for ModelState, ModelProfile, ProfileRegistry, ModelManager
# ---------------------------------------------------------------------------


@dataclass
class _FakeCapabilities:
    chat: bool = True
    completion: bool = True
    embeddings: bool = False
    tts: bool = False

    def to_dict(self) -> dict:
        return {
            "chat": self.chat,
            "completion": self.completion,
            "embeddings": self.embeddings,
            "tts": self.tts,
        }


@dataclass
class _FakeModelState:
    model_id: str
    display_name: str = "Test Model"
    backend_type: str = "vllm"
    backend_model_id: str = ""
    status: Any = None  # will be set in fixture
    load_policy: Any = None
    auto_reload: bool = False
    preferred_gpu: int | None = None
    current_gpu: list[int] = field(default_factory=list)
    vram_used_mb: int = 0
    capabilities: _FakeCapabilities = field(default_factory=_FakeCapabilities)
    last_request_at: Any = None
    loaded_at: Any = None
    worker_info: Any = None
    error_message: str | None = None
    extra_config: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.backend_model_id:
            parts = self.model_id.split("/", 1)
            self.backend_model_id = parts[-1] if "/" in self.model_id else self.model_id


@dataclass
class _FakeProfile:
    profile_id: str
    base_model_id: str
    display_name: str | None = None
    description: str | None = None
    category: str = "llm"
    load_overrides: dict | None = None
    request_defaults: dict | None = None
    assets: dict | None = None
    enabled: bool = True
    is_default: bool = False


class _FakeProfileRegistry:
    """In-memory registry for test purposes."""

    def __init__(self, profiles: list[_FakeProfile] | None = None):
        self._profiles: dict[str, _FakeProfile] = {}
        for p in profiles or []:
            self._profiles[p.profile_id] = p

    async def get(self, profile_id: str):
        return self._profiles.get(profile_id)

    async def list_by_model(self, base_model_id: str):
        return [p for p in self._profiles.values() if p.base_model_id == base_model_id]

    async def list_enabled(self):
        return [p for p in self._profiles.values() if p.enabled]

    async def list_all(self):
        return list(self._profiles.values())


class _FakeModelManager:
    """Simplified model manager for tests."""

    def __init__(self, states: dict[str, _FakeModelState] | None = None):
        self._states = dict(states or {})
        self._loaded: set[str] = set()

    async def get_state(self, model_id: str):
        return self._states.get(model_id)

    async def list_states(self):
        return list(self._states.values())

    async def load(self, model_id: str, force_gpu: int | None = None):
        state = self._states.get(model_id)
        if state is None:
            raise KeyError(f"Model '{model_id}' not configured")
        # Simulate loading
        from ocabra.core.model_manager import ModelStatus

        state.status = ModelStatus.LOADED
        self._loaded.add(model_id)
        return state

    async def unload(self, model_id: str, reason: str = "manual"):
        pass

    async def add_model(self, model_id: str, backend_type: str, **kwargs):
        from ocabra.core.model_manager import LoadPolicy, ModelStatus

        state = _FakeModelState(
            model_id=model_id,
            backend_type=backend_type,
            display_name=kwargs.get("display_name", model_id),
            status=ModelStatus.CONFIGURED,
            load_policy=LoadPolicy(kwargs.get("load_policy", "on_demand")),
            extra_config=kwargs.get("extra_config", {}),
        )
        self._states[model_id] = state
        return state

    async def touch_last_request_at(self, model_id: str, at):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_status():
    """Import and return ModelStatus enum."""
    from ocabra.core.model_manager import ModelStatus

    return ModelStatus


@pytest.fixture
def load_policy():
    from ocabra.core.model_manager import LoadPolicy

    return LoadPolicy


@pytest.fixture
def base_model_state(model_status, load_policy):
    return _FakeModelState(
        model_id="vllm/Qwen/Qwen3-8B",
        display_name="Qwen3-8B",
        backend_type="vllm",
        status=model_status.LOADED,
        load_policy=load_policy.ON_DEMAND,
    )


@pytest.fixture
def model_manager(base_model_state):
    return _FakeModelManager(states={"vllm/Qwen/Qwen3-8B": base_model_state})


@pytest.fixture
def enabled_profile():
    return _FakeProfile(
        profile_id="chat",
        base_model_id="vllm/Qwen/Qwen3-8B",
        display_name="Chat",
        enabled=True,
        is_default=True,
    )


@pytest.fixture
def disabled_profile():
    return _FakeProfile(
        profile_id="chat-disabled",
        base_model_id="vllm/Qwen/Qwen3-8B",
        display_name="Chat Disabled",
        enabled=False,
    )


# ---------------------------------------------------------------------------
# Resolución básica
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_valid_profile(model_manager, enabled_profile):
    """Resolver un profile_id válido y habilitado debe retornar (ModelProfile, ModelState)."""
    from ocabra.api.openai._deps import resolve_profile

    registry = _FakeProfileRegistry([enabled_profile])
    profile, state = await resolve_profile(
        "chat",
        model_manager,
        registry,
    )
    assert profile.profile_id == "chat"
    assert state.model_id == "vllm/Qwen/Qwen3-8B"


@pytest.mark.asyncio
async def test_resolve_disabled_profile_404(model_manager, disabled_profile):
    """Resolver un profile_id existente pero con enabled=False debe retornar 404."""
    from ocabra.api.openai._deps import resolve_profile

    registry = _FakeProfileRegistry([disabled_profile])
    with pytest.raises(HTTPException) as exc_info:
        await resolve_profile("chat-disabled", model_manager, registry)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_resolve_nonexistent_profile_404(model_manager):
    """Resolver un profile_id que no existe en BD debe retornar 404."""
    from ocabra.api.openai._deps import resolve_profile

    registry = _FakeProfileRegistry([])
    with pytest.raises(HTTPException) as exc_info:
        await resolve_profile("nonexistent", model_manager, registry)
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Legacy fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_fallback_enabled(model_manager, enabled_profile):
    """Con LEGACY_MODEL_ID_FALLBACK=true, un model_id canónico con '/' debe resolverse
    al perfil default del modelo correspondiente."""
    from ocabra.api.openai._deps import resolve_profile

    registry = _FakeProfileRegistry([enabled_profile])

    with patch("ocabra.config.settings") as mock_settings:
        mock_settings.legacy_model_id_fallback = True
        profile, state = await resolve_profile(
            "vllm/Qwen/Qwen3-8B",
            model_manager,
            registry,
        )
    assert profile.profile_id == "chat"
    assert state.model_id == "vllm/Qwen/Qwen3-8B"


@pytest.mark.asyncio
async def test_legacy_fallback_disabled(model_manager, enabled_profile):
    """Con LEGACY_MODEL_ID_FALLBACK=false, un model_id canónico con '/' debe retornar 404
    directamente, sin intentar buscar perfiles."""
    from ocabra.api.openai._deps import resolve_profile

    registry = _FakeProfileRegistry([enabled_profile])

    with patch("ocabra.config.settings") as mock_settings:
        mock_settings.legacy_model_id_fallback = False
        with pytest.raises(HTTPException) as exc_info:
            await resolve_profile("vllm/Qwen/Qwen3-8B", model_manager, registry)
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Merge de request_defaults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_defaults_merge():
    """Los request_defaults del perfil deben inyectarse como base en el body de la request."""
    from ocabra.api.openai._deps import merge_profile_defaults

    profile = _FakeProfile(
        profile_id="chat-creative",
        base_model_id="vllm/Qwen/Qwen3-8B",
        request_defaults={"temperature": 1.2, "top_p": 0.95},
    )

    body = {"model": "chat-creative", "messages": [{"role": "user", "content": "hi"}]}
    merged = merge_profile_defaults(profile, body)

    assert merged["temperature"] == 1.2
    assert merged["top_p"] == 0.95
    assert merged["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_request_defaults_client_overrides():
    """Los valores del body del cliente deben prevalecer sobre request_defaults del perfil."""
    from ocabra.api.openai._deps import merge_profile_defaults

    profile = _FakeProfile(
        profile_id="chat-creative",
        base_model_id="vllm/Qwen/Qwen3-8B",
        request_defaults={"temperature": 1.2, "top_p": 0.95},
    )

    body = {"model": "chat-creative", "temperature": 0.5}
    merged = merge_profile_defaults(profile, body)

    assert merged["temperature"] == 0.5  # client overrides
    assert merged["top_p"] == 0.95  # profile default preserved


# ---------------------------------------------------------------------------
# Assets injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assets_injection_voice_ref():
    """Un perfil TTS con assets voice_ref debe inyectar el path en el body forwarded."""
    from ocabra.api.openai._deps import merge_profile_defaults

    profile = _FakeProfile(
        profile_id="tts-glados",
        base_model_id="tts/chatterbox",
        category="tts",
        assets={
            "voice_ref": {
                "filename": "reference.wav",
                "path": "/data/profiles/tts-glados/reference.wav",
                "size_bytes": 12345,
            }
        },
    )

    # Client tries to set voice_ref — should be overridden by asset
    body = {"model": "tts-glados", "input": "Hello", "voice_ref": "/evil/path"}
    merged = merge_profile_defaults(profile, body)

    # Asset-injected path wins over client value
    assert merged["voice_ref"] == "/data/profiles/tts-glados/reference.wav"


@pytest.mark.asyncio
async def test_assets_injection_voice_ref_string():
    """Assets with simple string voice_ref are also supported."""
    from ocabra.api.openai._deps import merge_profile_defaults

    profile = _FakeProfile(
        profile_id="tts-custom",
        base_model_id="tts/kokoro",
        category="tts",
        assets={"voice_ref": "/data/profiles/tts-custom/voice.wav"},
    )

    body = {"model": "tts-custom", "input": "Hello"}
    merged = merge_profile_defaults(profile, body)
    assert merged["voice_ref"] == "/data/profiles/tts-custom/voice.wav"


# ---------------------------------------------------------------------------
# Worker key: compartido vs dedicado
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_key_shared():
    """Dos perfiles del mismo modelo con load_overrides idénticos deben compartir worker."""
    from ocabra.api.openai._deps import compute_worker_key

    overrides = {"max_model_len": 8192}
    key1 = compute_worker_key("vllm/Qwen/Qwen3-8B", overrides)
    key2 = compute_worker_key("vllm/Qwen/Qwen3-8B", overrides)
    assert key1 == key2
    # With overrides, the key should differ from the base
    assert key1 != "vllm/Qwen/Qwen3-8B"


@pytest.mark.asyncio
async def test_worker_key_dedicated():
    """Dos perfiles del mismo modelo con load_overrides diferentes deben usar workers separados."""
    from ocabra.api.openai._deps import compute_worker_key

    key_short = compute_worker_key("vllm/Qwen/Qwen3-8B", {"max_model_len": 8192})
    key_long = compute_worker_key("vllm/Qwen/Qwen3-8B", {"max_model_len": 32768})
    assert key_short != key_long


@pytest.mark.asyncio
async def test_worker_key_empty_overrides():
    """Empty or None load_overrides should return base_model_id as the key."""
    from ocabra.api.openai._deps import compute_worker_key

    assert compute_worker_key("vllm/Qwen/Qwen3-8B", None) == "vllm/Qwen/Qwen3-8B"
    assert compute_worker_key("vllm/Qwen/Qwen3-8B", {}) == "vllm/Qwen/Qwen3-8B"


@pytest.mark.asyncio
async def test_worker_key_deterministic():
    """The same overrides in different order should produce the same key."""
    from ocabra.api.openai._deps import compute_worker_key

    key1 = compute_worker_key("vllm/model", {"a": 1, "b": 2})
    key2 = compute_worker_key("vllm/model", {"b": 2, "a": 1})
    assert key1 == key2


# ---------------------------------------------------------------------------
# Ollama profile resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_resolve_profile_first(model_manager, enabled_profile):
    """Ollama endpoints should resolve by profile_id before trying legacy model names."""
    from ocabra.api.openai._deps import resolve_profile

    registry = _FakeProfileRegistry([enabled_profile])
    profile, state = await resolve_profile(
        "chat",
        model_manager,
        registry,
    )
    assert profile.profile_id == "chat"
    assert state.model_id == "vllm/Qwen/Qwen3-8B"


@pytest.mark.asyncio
async def test_ollama_resolve_profile_with_user_access(model_manager, enabled_profile):
    """Profile resolution respects user access control."""
    from ocabra.api.openai._deps import resolve_profile

    registry = _FakeProfileRegistry([enabled_profile])

    # User with access to this profile
    user_ok = type("UserContext", (), {
        "is_admin": False,
        "accessible_model_ids": {"chat"},
    })()
    profile, state = await resolve_profile(
        "chat", model_manager, registry, user=user_ok
    )
    assert profile.profile_id == "chat"

    # User without access
    user_no = type("UserContext", (), {
        "is_admin": False,
        "accessible_model_ids": set(),
    })()
    with pytest.raises(Exception) as exc_info:
        await resolve_profile("chat", model_manager, registry, user=user_no)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_ollama_resolve_profile_admin_bypass(model_manager, enabled_profile):
    """Admin users can access any profile regardless of group membership."""
    from ocabra.api.openai._deps import resolve_profile

    registry = _FakeProfileRegistry([enabled_profile])

    admin_user = type("UserContext", (), {
        "is_admin": True,
        "accessible_model_ids": set(),
    })()
    profile, state = await resolve_profile(
        "chat", model_manager, registry, user=admin_user
    )
    assert profile.profile_id == "chat"


# ---------------------------------------------------------------------------
# Chatterbox voice_ref path validation
# ---------------------------------------------------------------------------


class TestVoiceRefPathValidation:
    """Test that chatterbox worker rejects voice_ref paths outside controlled dirs."""

    def test_safe_data_profiles_path(self):
        from workers.chatterbox_worker import _is_voice_ref_path_safe

        assert _is_voice_ref_path_safe("/data/profiles/tts-glados/reference.wav")

    def test_safe_data_models_path(self):
        from workers.chatterbox_worker import _is_voice_ref_path_safe

        assert _is_voice_ref_path_safe("/data/models/some-model/audio.wav")

    def test_safe_tmp_path(self):
        from workers.chatterbox_worker import _is_voice_ref_path_safe

        assert _is_voice_ref_path_safe("/tmp/decoded-audio.wav")

    def test_reject_etc_passwd(self):
        from workers.chatterbox_worker import _is_voice_ref_path_safe

        assert not _is_voice_ref_path_safe("/etc/passwd")

    def test_reject_traversal(self):
        from workers.chatterbox_worker import _is_voice_ref_path_safe

        assert not _is_voice_ref_path_safe("/data/profiles/../../../etc/passwd")

    def test_reject_home_dir(self):
        from workers.chatterbox_worker import _is_voice_ref_path_safe

        assert not _is_voice_ref_path_safe("/home/user/evil.wav")

    def test_reject_empty(self):
        from workers.chatterbox_worker import _is_voice_ref_path_safe

        assert not _is_voice_ref_path_safe("")

    def test_resolve_voice_ref_rejects_unsafe_path(self):
        from workers.chatterbox_worker import _resolve_voice_ref

        result = _resolve_voice_ref("/etc/passwd", None)
        assert result is None

    def test_resolve_voice_ref_accepts_safe_nonexistent(self):
        """Safe path that doesn't exist on disk returns None (no file)."""
        from workers.chatterbox_worker import _resolve_voice_ref

        result = _resolve_voice_ref("/data/profiles/nonexistent.wav", None)
        assert result is None


# ---------------------------------------------------------------------------
# Local scanner Chatterbox detection
# ---------------------------------------------------------------------------


class TestLocalScannerBackendDetection:
    """Test that local_scanner detects model backends correctly."""

    def test_detect_chatterbox(self, tmp_path):
        from ocabra.registry.local_scanner import LocalScanner

        scanner = LocalScanner()
        model_dir = tmp_path / "ResembleAI--chatterbox-turbo"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        assert scanner._detect_hf_backend(model_dir) == "chatterbox"

    def test_detect_whisper(self, tmp_path):
        from ocabra.registry.local_scanner import LocalScanner

        scanner = LocalScanner()
        model_dir = tmp_path / "openai--whisper-large-v3"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "whisper"}')
        assert scanner._detect_hf_backend(model_dir) == "whisper"

    def test_detect_diffusers(self, tmp_path):
        from ocabra.registry.local_scanner import LocalScanner

        scanner = LocalScanner()
        model_dir = tmp_path / "stable-diffusion-v1-5"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model_index.json").write_text("{}")
        assert scanner._detect_hf_backend(model_dir) == "diffusers"

    def test_detect_default_vllm(self, tmp_path):
        from ocabra.registry.local_scanner import LocalScanner

        scanner = LocalScanner()
        model_dir = tmp_path / "Qwen--Qwen3-8B"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        assert scanner._detect_hf_backend(model_dir) == "vllm"
