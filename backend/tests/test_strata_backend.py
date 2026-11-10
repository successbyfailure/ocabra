from __future__ import annotations

import json
import signal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ocabra.api.internal.models import _apply_capability_fallbacks
from ocabra.backends.strata_backend import StrataBackend
from ocabra.core import artifacts
from ocabra.core.model_manager import ModelState, ModelStatus, declared_capabilities


class _Process:
    pid = 4321
    returncode = None

    async def wait(self):
        self.returncode = -signal.SIGINT
        return self.returncode


def _runtime(tmp_path: Path) -> tuple[Path, Path]:
    runtime = tmp_path / "runtime" / "terminal" / "0.1.59.1"
    python = runtime / "python" / "bin" / "python3.12"
    server = runtime / "serve" / "server.py"
    python.parent.mkdir(parents=True)
    server.parent.mkdir(parents=True)
    python.touch()
    server.touch()
    config = tmp_path / "strata-coder.json"
    config.write_text(
        json.dumps(
            {
                "exe": "component:strata-engine",
                "args": ["--max-context", "65536", "--kv", "int8"],
                "cwd": str(runtime),
                "model_name": "qwen3.8-coder-iq1_m",
                "gpu": [0, 1],
                "layer_split": "14",
                "parallel": 2,
            }
        )
    )
    return runtime, config


@pytest.mark.asyncio
async def test_load_private_config_capabilities_and_cleanup(tmp_path, monkeypatch) -> None:
    runtime, config = _runtime(tmp_path)
    process = _Process()
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
    backend = StrataBackend()
    monkeypatch.setattr(backend, "_wait_for_startup", AsyncMock())
    extra = {
        "strata": {
            "config_path": str(config),
            "runtime_dir": str(runtime),
            "native_context_length": 262_144,
        },
        "_runtime": {"capacity_vram_by_gpu_mb": {"0": 11_297, "1": 23_696}},
    }
    info = await backend.load("coder", [0, 1], port=18501, extra_config=extra)
    worker = backend._processes["coder"]
    private = json.loads(worker.temp_config_path.read_text())
    assert info.vram_used_mb == 34_993
    assert private["host"] == "127.0.0.1" and private["api_key"]
    assert spawn.await_args.kwargs["start_new_session"] is True
    caps = await backend.get_capabilities("coder")
    assert caps.chat and caps.tools and caps.reasoning and caps.streaming
    assert not caps.completion
    assert (caps.context_length, caps.parallel_slots, caps.context_per_request) == (
        65_536,
        2,
        65_536,
    )
    assert backend.request_headers("coder") == {"Authorization": f"Bearer {private['api_key']}"}
    monkeypatch.setattr("os.killpg", lambda *_args: None)
    await backend.unload("coder")
    assert not worker.temp_config_path.exists()


@pytest.mark.asyncio
async def test_load_rejects_different_gpu_order(tmp_path) -> None:
    runtime, config = _runtime(tmp_path)
    with pytest.raises(ValueError, match="targets GPUs"):
        await StrataBackend().load(
            "coder",
            [1, 0],
            port=18501,
            extra_config={"strata": {"config_path": str(config), "runtime_dir": str(runtime)}},
        )


def test_strata_artifact_materialization(tmp_path) -> None:
    path = tmp_path / "strata.json"
    path.touch()
    artifact = {"kind": "strata_config", "source": "local", "ref": "coder", "path": str(path)}
    extra = artifacts.materialize("strata", artifact, {"strata": {"startup_timeout_s": 300}})
    assert extra["strata"]["config_path"] == str(path)
    assert "config_path" not in artifacts.strip_derived("strata", extra)["strata"]
    assert artifacts.artifact_from_legacy_extra("strata", "coder", extra) == artifact


def test_cold_capabilities_use_strata_shape() -> None:
    state = type(
        "State",
        (),
        {
            "backend_type": "strata",
            "status": ModelStatus.CONFIGURED,
            "extra_config": {
                "strata": {"context_length": 65_536, "native_context_length": 262_144},
                "_runtime": {"parallelism": 2},
            },
        },
    )()
    caps = _apply_capability_fallbacks(state, {"chat": True, "parallel_slots": 1})
    assert (caps["context_length"], caps["parallel_slots"], caps["context_per_request"]) == (
        65_536,
        2,
        65_536,
    )


def test_declared_capabilities_use_strata_shape() -> None:
    state = ModelState(
        model_id="strata/coder",
        display_name="Strata Coder",
        backend_type="strata",
        backend_model_id="coder",
        status=ModelStatus.CONFIGURED,
        extra_config={
            "strata": {
                "context_length": 65_536,
                "native_context_length": 262_144,
                "parallel_slots": 2,
            },
            "_runtime": {"parallelism": 2},
        },
    )
    caps = declared_capabilities(state)
    assert caps.chat and caps.tools and caps.reasoning and caps.streaming
    assert not caps.completion
    assert (caps.context_length, caps.native_context_length) == (65_536, 262_144)
    assert (caps.parallel_slots, caps.context_per_request) == (2, 65_536)
