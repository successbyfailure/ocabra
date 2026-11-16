"""Pulling an Ollama tag straight from the registry (no daemon on cloud nodes)."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from ocabra.registry.ollama_registry import OllamaRegistry

MODEL = b"GGUF" + b"\x01" * 5000
PROJ = b"GGUF" + b"\x02" * 300
CONFIG = b'{"model_format":"gguf"}'


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


MANIFEST = {
    "schemaVersion": 2,
    "config": {"digest": _digest(CONFIG), "size": len(CONFIG)},
    "layers": [
        {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": _digest(MODEL),
            "size": len(MODEL),
        },
        {
            "mediaType": "application/vnd.ollama.image.projector",
            "digest": _digest(PROJ),
            "size": len(PROJ),
        },
    ],
}
BLOBS = {_digest(MODEL): MODEL, _digest(PROJ): PROJ, _digest(CONFIG): CONFIG}


def _registry(corrupt: bool = False, seen: list | None = None):
    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if seen is not None:
            seen.append(path)
        assert request.url.host == "registry.ollama.ai"
        if path == "/v2/library/gemma4/manifests/26b":
            return httpx.Response(200, content=json.dumps(MANIFEST).encode())
        if path.startswith("/v2/library/gemma4/blobs/"):
            digest = path.rsplit("/", 1)[1]
            data = BLOBS[digest]
            return httpx.Response(200, content=(b"X" + data[1:]) if corrupt else data)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def test_pull_from_registry_writes_ollama_store_layout(tmp_path):
    progress = []
    await OllamaRegistry("http://nowhere").pull_from_registry(
        "gemma4:26b", lambda pct, speed: progress.append(pct), root=tmp_path, transport=_registry()
    )
    manifest = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "gemma4" / "26b"
    assert json.loads(manifest.read_text()) == MANIFEST
    for digest, data in BLOBS.items():
        assert (tmp_path / "blobs" / digest.replace(":", "-")).read_bytes() == data
    assert progress[-1] == 100.0
    assert not list((tmp_path / "blobs").glob("*.partial"))

    # What llama.cpp-on-Ollama-tags reads back resolves to those files.
    from ocabra.core.artifacts import read_ollama_manifest

    layers = read_ollama_manifest("gemma4:26b", tmp_path)
    assert layers["model"] == _digest(MODEL).replace(":", "-")


async def test_existing_blobs_are_not_downloaded_again(tmp_path):
    reg = OllamaRegistry("http://nowhere")
    await reg.pull_from_registry(
        "gemma4:26b", lambda *a: None, root=tmp_path, transport=_registry()
    )
    seen: list[str] = []
    await reg.pull_from_registry(
        "gemma4:26b", lambda *a: None, root=tmp_path, transport=_registry(seen=seen)
    )
    assert seen == ["/v2/library/gemma4/manifests/26b"]


async def test_corrupt_blob_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        await OllamaRegistry("http://nowhere").pull_from_registry(
            "gemma4:26b", lambda *a: None, root=tmp_path, transport=_registry(corrupt=True)
        )
    assert not list((tmp_path / "blobs").glob("sha256-*[!l]"))  # nothing kept
    assert not (tmp_path / "manifests").exists()


async def test_pull_falls_back_to_registry_without_daemon(monkeypatch, tmp_path):
    reg = OllamaRegistry("http://ollama:11434")
    called = {}

    async def no_daemon(model_ref, cb):
        raise httpx.ConnectError("Name or service not known")

    async def from_registry(model_ref, cb, **kw):
        called["ref"] = model_ref
        return tmp_path

    monkeypatch.setattr(reg, "_pull_via_daemon", no_daemon)
    monkeypatch.setattr(reg, "pull_from_registry", from_registry)
    await reg.pull("gemma4:26b", lambda *a: None)
    assert called == {"ref": "gemma4:26b"}
