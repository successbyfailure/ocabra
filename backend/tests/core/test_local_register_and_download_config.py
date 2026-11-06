"""Registering on-disk weights, and the register_config casing bug (2026-11-06)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from ocabra.api.internal.downloads import _rc_get
from ocabra.api.internal.registry import RegisterLocalModelRequest, register_local_model


class TestRegisterConfigCasing:
    """``api/client.ts`` shipped this payload with a plain JSON.stringify — no
    case conversion — so it arrived camelCased while every reader expected
    snake_case. Every field was silently discarded: the load policy and display
    name chosen in the Explore install modal, and `extraConfig` carrying the
    whole vLLM recipe preconfiguration (verified against the DB: zero
    registered vLLM models carried a recipeId). The client now maps the keys,
    and this reader accepts both so older clients are fixed too.
    """

    def test_accepts_snake_case(self) -> None:
        assert _rc_get({"load_policy": "warm"}, "load_policy", "on_demand") == "warm"

    def test_accepts_camel_case(self) -> None:
        assert _rc_get({"loadPolicy": "pin"}, "load_policy", "on_demand") == "pin"

    def test_snake_wins_when_both_present(self) -> None:
        payload = {"load_policy": "warm", "loadPolicy": "pin"}
        assert _rc_get(payload, "load_policy", "on_demand") == "warm"

    def test_falls_back_to_default(self) -> None:
        assert _rc_get({}, "load_policy", "on_demand") == "on_demand"

    def test_carries_nested_extra_config(self) -> None:
        recipe = {"vllm": {"recipeId": "llama-3"}}
        assert _rc_get({"extraConfig": recipe}, "extra_config") == recipe


def _request(models_dir: str, *, existing: bool = False) -> SimpleNamespace:
    mm = SimpleNamespace(
        get_state=AsyncMock(return_value=object() if existing else None),
        add_model=AsyncMock(
            return_value=SimpleNamespace(model_id="x", to_dict=lambda: {"model_id": "x"})
        ),
    )
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(model_manager=mm, profile_registry=None))
    )


class TestRegisterLocalModel:
    @pytest.mark.asyncio
    async def test_rejects_unknown_backend(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("ocabra.api.internal.registry.settings.models_dir", str(tmp_path))
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        body = RegisterLocalModelRequest(
            model_ref="m", path=str(gguf), backend_type="not_a_backend"
        )
        with pytest.raises(HTTPException) as exc:
            await register_local_model(body, _request(str(tmp_path)))
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_refuses_paths_outside_the_configured_roots(self, tmp_path, monkeypatch) -> None:
        """This endpoint takes a filesystem path from an API caller. Without the
        containment check it would point a worker at any readable host file."""
        models = tmp_path / "models"
        models.mkdir()
        outside = tmp_path / "outside.gguf"
        outside.write_bytes(b"x")
        monkeypatch.setattr("ocabra.api.internal.registry.settings.models_dir", str(models))
        monkeypatch.setattr("ocabra.api.internal.registry.settings.ollama_shared_models_dir", "")
        body = RegisterLocalModelRequest(
            model_ref="m", path=str(outside), backend_type="llamacpp_mks"
        )
        with pytest.raises(HTTPException) as exc:
            await register_local_model(body, _request(str(models)))
        assert exc.value.status_code == 400
        assert "outside the configured roots" in str(exc.value.detail)

    @pytest.mark.asyncio
    async def test_accepts_an_ollama_blob_and_pins_the_exact_path(
        self, tmp_path, monkeypatch
    ) -> None:
        """The whole point of the Ollama case: the blob filename is a sha256
        digest unrelated to the model name, so the backend's directory/stem
        heuristics cannot find it. The path must be pinned in extra_config."""
        models = tmp_path / "models"
        models.mkdir()
        ollama = tmp_path / "ollama"
        (ollama / "blobs").mkdir(parents=True)
        blob = ollama / "blobs" / "sha256-abc123"
        blob.write_bytes(b"GGUF")
        monkeypatch.setattr("ocabra.api.internal.registry.settings.models_dir", str(models))
        monkeypatch.setattr(
            "ocabra.api.internal.registry.settings.ollama_shared_models_dir", str(ollama)
        )
        req = _request(str(models))
        body = RegisterLocalModelRequest(
            model_ref="gemma4:26b", path=str(blob), backend_type="llamacpp_mks"
        )

        await register_local_model(body, req)

        kwargs = req.app.state.model_manager.add_model.await_args.kwargs
        assert kwargs["model_id"] == "llamacpp_mks/gemma4:26b"
        assert kwargs["backend_type"] == "llamacpp_mks"
        # Shared section name: the same config works on either llama.cpp backend.
        assert kwargs["extra_config"] == {"llama_cpp": {"model_path": str(blob)}}

    @pytest.mark.asyncio
    async def test_conflict_when_already_registered(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("ocabra.api.internal.registry.settings.models_dir", str(tmp_path))
        monkeypatch.setattr("ocabra.api.internal.registry.settings.ollama_shared_models_dir", "")
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        body = RegisterLocalModelRequest(model_ref="m", path=str(gguf), backend_type="llamacpp_mks")
        with pytest.raises(HTTPException) as exc:
            await register_local_model(body, _request(str(tmp_path), existing=True))
        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_missing_file_is_404(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("ocabra.api.internal.registry.settings.models_dir", str(tmp_path))
        monkeypatch.setattr("ocabra.api.internal.registry.settings.ollama_shared_models_dir", "")
        body = RegisterLocalModelRequest(
            model_ref="m", path=str(tmp_path / "nope.gguf"), backend_type="llamacpp_mks"
        )
        with pytest.raises(HTTPException) as exc:
            await register_local_model(body, _request(str(tmp_path)))
        assert exc.value.status_code == 404
