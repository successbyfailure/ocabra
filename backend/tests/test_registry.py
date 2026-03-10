import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import AsyncClient

from ocabra.registry.huggingface import HuggingFaceRegistry
from ocabra.registry.local_scanner import LocalScanner
from ocabra.registry.ollama_registry import OllamaRegistry


@pytest.mark.asyncio
async def test_hf_search_with_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_list_models(**kwargs):
        captured.update(kwargs)
        return [
            SimpleNamespace(
                id="meta-llama/Llama-3.2-3B-Instruct",
                pipeline_tag="text-generation",
                downloads=123,
                likes=45,
                tags=["llama", "chat"],
                gated=False,
            )
        ]

    monkeypatch.setattr("ocabra.registry.huggingface.list_models", fake_list_models)

    registry = HuggingFaceRegistry()
    cards = await registry.search(query="llama", task="text-generation", limit=5)

    assert len(cards) == 1
    assert cards[0].repo_id == "meta-llama/Llama-3.2-3B-Instruct"
    assert cards[0].task == "text-generation"
    assert captured["search"] == "llama"
    assert captured["task"] == "text-generation"
    assert captured["limit"] == 5


@pytest.mark.asyncio
async def test_hf_detail_infers_vllm_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = SimpleNamespace(
        id="meta-llama/Llama-3.2-3B-Instruct",
        pipeline_tag="text-generation",
        downloads=1000,
        likes=200,
        tags=["text-generation"],
        gated=False,
        siblings=[SimpleNamespace(rfilename="model.safetensors", size=4 * 1024**3)],
    )

    monkeypatch.setattr("ocabra.registry.huggingface.model_info", lambda **_: fake_info)

    registry = HuggingFaceRegistry()
    detail = await registry.get_model_detail("meta-llama/Llama-3.2-3B-Instruct")

    assert detail.suggested_backend == "vllm"
    assert detail.estimated_vram_gb is not None
    assert detail.estimated_vram_gb > 5.0


@pytest.mark.asyncio
async def test_ollama_pull_parses_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStdout:
        def __init__(self) -> None:
            self._lines = [
                json.dumps({"status": "pulling", "completed": 50, "total": 100}).encode() + b"\n",
                json.dumps({"status": "pulling", "completed": 100, "total": 100}).encode() + b"\n",
                b"",
            ]

        async def readline(self) -> bytes:
            await asyncio.sleep(0)
            return self._lines.pop(0)

    class FakeProc:
        def __init__(self) -> None:
            self.stdout = FakeStdout()

        async def wait(self) -> int:
            return 0

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return FakeProc()

    monkeypatch.setattr("ocabra.registry.ollama_registry.shutil.which", lambda _cmd: "/usr/bin/ollama")
    monkeypatch.setattr(
        "ocabra.registry.ollama_registry.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    updates: list[float] = []
    registry = OllamaRegistry()
    await registry.pull("llama3.2:3b", lambda pct, _speed: updates.append(pct))

    assert updates
    assert any(pct >= 50 for pct in updates)
    assert updates[-1] == 100.0


@pytest.mark.asyncio
async def test_local_scanner_discovers_hf_gguf_and_ollama(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf-model"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "weights.safetensors").write_bytes(b"1234")

    gguf_file = tmp_path / "tiny.gguf"
    gguf_file.write_bytes(b"abcd")

    ollama_dir = tmp_path / "ollama-model"
    ollama_dir.mkdir()
    (ollama_dir / "Modelfile").write_text("FROM llama3.2", encoding="utf-8")

    scanner = LocalScanner()
    models = await scanner.scan(tmp_path)

    sources = {model.source for model in models}
    assert sources == {"huggingface", "gguf", "ollama"}
    assert any(model.model_ref == "hf-model" for model in models)
    assert any(model.model_ref == "tiny" for model in models)
    assert any(model.model_ref == "ollama-model" for model in models)


@pytest.mark.asyncio
async def test_registry_hf_endpoint(async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(query: str, task: str | None, limit: int):
        _ = query, task, limit
        return [
            {
                "repo_id": "org/model",
                "model_name": "model",
                "task": "text-generation",
                "downloads": 1,
                "likes": 2,
                "size_gb": None,
                "tags": [],
                "gated": False,
            }
        ]

    monkeypatch.setattr("ocabra.api.internal.registry._hf_registry.search", fake_search)
    response = await async_client.get("/ocabra/registry/hf", params={"q": "model"})

    assert response.status_code == 200
    assert response.json()[0]["repo_id"] == "org/model"


@pytest.mark.asyncio
async def test_registry_ollama_endpoint(async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(query: str):
        _ = query
        return [
            {
                "name": "llama3.2:3b",
                "description": "Test",
                "tags": ["latest"],
                "size_gb": 1.2,
                "pulls": 100,
            }
        ]

    monkeypatch.setattr("ocabra.api.internal.registry._ollama_registry.search", fake_search)
    response = await async_client.get("/ocabra/registry/ollama", params={"q": "llama"})

    assert response.status_code == 200
    assert response.json()[0]["name"] == "llama3.2:3b"
