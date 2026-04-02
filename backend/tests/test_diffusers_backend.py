import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ocabra.backends.base import WorkerInfo
from ocabra.backends.diffusers_backend import DiffusersBackend


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self) -> dict:
        return self._payload


class DummyAsyncClient:
    def __init__(self, recorder: dict) -> None:
        self._recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url: str, json: dict) -> DummyResponse:
        self._recorder["url"] = url
        self._recorder["json"] = json
        return DummyResponse({"images": [{"b64_json": "abc123"}]})


class DummyProcess:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.returncode = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.returncode = 0
        return 0


@pytest.mark.asyncio
async def test_load_and_unload_with_subprocess_mock(tmp_path):
    models_dir = tmp_path / "models"
    model_id = "sdxl-test"
    (models_dir / model_id).mkdir(parents=True)

    backend = DiffusersBackend()
    fake_process = DummyProcess()

    with (
        patch("ocabra.backends.diffusers_backend.settings.models_dir", str(models_dir)),
        patch.object(backend, "_assign_port", new=AsyncMock(return_value=18077)),
        patch.object(backend, "_wait_until_healthy", new=AsyncMock(return_value=None)),
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake_process)) as spawn,
    ):
        info = await backend.load(model_id, [1])

        assert info.backend_type == "diffusers"
        assert info.model_id == model_id
        assert info.port == 18077
        assert info.pid == fake_process.pid
        assert backend._workers[model_id] == info
        assert spawn.await_count == 1

        await backend.unload(model_id)
        assert fake_process.terminated is True


@pytest.mark.asyncio
async def test_forward_request_translates_openai_size():
    backend = DiffusersBackend()
    backend._workers["flux-1"] = WorkerInfo(
        backend_type="diffusers",
        model_id="flux-1",
        gpu_indices=[1],
        port=18123,
        pid=9001,
        vram_used_mb=12000,
    )

    recorder: dict = {}

    def _client_factory(*args, **kwargs):
        return DummyAsyncClient(recorder)

    body = {
        "prompt": "a cat in space",
        "n": 2,
        "size": "1792x1024",
        "response_format": "b64_json",
    }

    with patch("httpx.AsyncClient", new=_client_factory):
        response = await backend.forward_request("flux-1", "/v1/images/generations", body)

    assert recorder["url"].endswith("/generate")
    assert recorder["json"]["width"] == 1792
    assert recorder["json"]["height"] == 1024
    assert recorder["json"]["num_images"] == 2
    assert response["data"] == [{"b64_json": "abc123"}]


def test_detect_pipeline_class_from_model_index(tmp_path):
    worker_path = Path(__file__).resolve().parents[1] / "workers" / "diffusers_worker.py"
    spec = importlib.util.spec_from_file_location("diffusers_worker_module", worker_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_dir = tmp_path / "flux-model"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "FluxPipeline"}), encoding="utf-8"
    )

    pipeline_type = module.detect_pipeline_class(model_dir)
    assert pipeline_type == "FluxPipeline"
