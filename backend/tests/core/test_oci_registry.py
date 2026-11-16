"""Prebuilt backends: pack, push, pull and install (core/oci_registry.py).

A fake registry (httpx.MockTransport) speaks the real protocol — token
challenge, chunked blob uploads, manifests by tag — so the client is exercised
end to end without network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import httpx
import pytest

import ocabra.core.oci_registry as oci
from ocabra.core.backend_installer import BackendInstaller, BackendInstallStatus
from ocabra.core.worker_pool import WorkerPool
from tests.core.test_backend_installer import _InstallableMockBackend

REPO = "ghcr.test/acme/ocabra-backends"


class FakeRegistry:
    """Just enough of the OCI distribution API, with a pull/push token dance."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.manifests: dict[str, bytes] = {}
        self.uploads: dict[str, bytearray] = {}
        self.patches = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        if url.path == "/token":
            return httpx.Response(
                200, json={"token": "push" if request.headers.get("authorization") else "pull"}
            )
        if request.headers.get("authorization") not in ("Bearer pull", "Bearer push"):
            return httpx.Response(
                401,
                headers={
                    "www-authenticate": 'Bearer realm="https://ghcr.test/token",service="ghcr.test"'
                },
            )
        writing = request.method in ("POST", "PATCH", "PUT")
        if writing and request.headers["authorization"] != "Bearer push":
            return httpx.Response(403)
        path = url.path
        m = re.match(r"^/v2/(.+)/(manifests|blobs)/(.+)$", path)
        if path.endswith("/blobs/uploads/") and request.method == "POST":
            uid = f"u{len(self.uploads)}"
            self.uploads[uid] = bytearray()
            return httpx.Response(202, headers={"location": f"/v2/upload/{uid}"})
        if path.startswith("/v2/upload/"):
            uid = path.rsplit("/", 1)[1]
            if request.method == "PATCH":
                # Like GHCR: ranged chunks over 4 MiB are refused; a plain
                # stream (what we send) is fine.
                assert "content-range" not in request.headers
                self.uploads[uid] += request.read()
                self.patches += 1
                return httpx.Response(202, headers={"location": f"/v2/upload/{uid}"})
            data = bytes(self.uploads.pop(uid))
            digest = url.params["digest"]
            assert digest == "sha256:" + hashlib.sha256(data).hexdigest()
            self.blobs[digest] = data
            return httpx.Response(201)
        assert m, path
        kind, ref = m.group(2), m.group(3)
        if kind == "blobs":
            if ref not in self.blobs:
                return httpx.Response(404)
            body = b"" if request.method == "HEAD" else self.blobs[ref]
            return httpx.Response(200, content=body)
        if request.method == "PUT":
            self.manifests[ref] = request.content
            return httpx.Response(201)
        if ref not in self.manifests and not any(
            "sha256:" + hashlib.sha256(v).hexdigest() == ref for v in self.manifests.values()
        ):
            return httpx.Response(404)
        body = self.manifests.get(ref) or next(
            v for v in self.manifests.values() if "sha256:" + hashlib.sha256(v).hexdigest() == ref
        )
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        return httpx.Response(
            200,
            content=b"" if request.method == "HEAD" else body,
            headers={"docker-content-digest": digest, "content-type": oci.OCI_MANIFEST},
        )


@pytest.fixture
def registry(monkeypatch):
    """Route every RegistryClient to the fake registry."""
    fake = FakeRegistry()
    real_init = oci.RegistryClient.__init__

    def _init(self, ref, *, username=None, password=None, client=None, pull_only=False):
        real_init(
            self,
            ref,
            username=username,
            password=password,
            client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)),
            pull_only=pull_only,
        )

    monkeypatch.setattr(oci.RegistryClient, "__init__", _init)
    monkeypatch.setattr(oci, "UPLOAD_CHUNK", 1000)  # the stream comes in many pieces
    return fake


def _backend_tree(root: Path, target: Path, marker: str = "v1") -> Path:
    """A backend folder like the installer leaves, built for *target*."""
    root.mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "bin" / "tool").write_text(f"#!/bin/sh\necho {marker}\n")
    os.chmod(root / "bin" / "tool", 0o755)
    (root / "lib").mkdir()
    (root / "lib" / "big.bin").write_bytes(os.urandom(5000))
    (root / "lib" / "hard.bin").write_bytes(b"same")
    os.link(root / "lib" / "hard.bin", root / "lib" / "hard-link.bin")
    os.symlink("/usr/bin/env", root / "bin" / "abs-link")  # venv-style absolute link
    (root / "empty").mkdir()
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "backend_type": "mock",
                "version": marker,
                "install_source": "source",
                "python_bin": None,
                "extra_bins": {"tool": str(target / "bin" / "tool")},
                "pip_packages": ["cowsay==6.1"],
            }
        )
    )
    return root


def test_pack_and_unpack_round_trip(tmp_path):
    tree = _backend_tree(tmp_path / "src", tmp_path / "target")
    layers = oci.pack_tree(tree, tmp_path / "out", layer_bytes=3000)
    assert len(layers) >= 2  # split by size
    out = oci.unpack_layers([lyr.path for lyr in layers], tmp_path / "x")
    assert (out / "bin" / "tool").read_text().endswith("v1\n")
    assert os.access(out / "bin" / "tool", os.X_OK)
    assert (out / "lib" / "big.bin").read_bytes() == (tree / "lib" / "big.bin").read_bytes()
    # Hard links come out as plain copies (they may land in different layers).
    assert (out / "lib" / "hard-link.bin").read_bytes() == b"same"
    assert os.readlink(out / "bin" / "abs-link") == "/usr/bin/env"
    assert (out / "empty").is_dir()
    for lyr in layers:  # digests are what a registry will check
        assert lyr.digest == "sha256:" + hashlib.sha256(lyr.path.read_bytes()).hexdigest()


def test_packing_is_deterministic(tmp_path):
    """Republishing an unchanged backend must reuse the blobs, not re-upload GBs."""
    import time

    tree = _backend_tree(tmp_path / "src", tmp_path / "target")
    first = oci.pack_tree(tree, tmp_path / "a", layer_bytes=3000)
    time.sleep(1.1)
    (tree / "lib" / "big.bin").read_bytes()  # touch atime
    second = oci.pack_tree(tree, tmp_path / "b", layer_bytes=3000)
    assert [x.digest for x in first] == [x.digest for x in second]


def test_unpack_refuses_paths_escaping_the_destination(tmp_path):
    import gzip
    import io
    import tarfile

    evil = tmp_path / "evil.tar.gz"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("backend/../../escaped")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    evil.write_bytes(gzip.compress(buf.getvalue()))
    with pytest.raises(tarfile.TarError):
        oci.unpack_layers([evil], tmp_path / "dest")
    assert not (tmp_path / "escaped").exists()


async def test_publish_then_inspect_then_pull(registry, tmp_path):
    tree = _backend_tree(tmp_path / "src", tmp_path / "target")
    out = await oci.publish_backend(tree, "mock", REPO, password="tok", log=lambda _m: None)
    assert out["ref"] == f"{REPO}:mock" and re.search(
        r":mock-\d{8}-[0-9a-f]{10}$", out["immutable"]
    )
    assert registry.patches >= 2  # one streamed PATCH per blob (config + layers)
    assert not list(tmp_path.glob(".oci-publish-*"))  # work dir cleaned

    found = await oci.inspect_backend(REPO, "mock")
    assert found is not None and found.digest == out["digest"]
    assert found.summary()["version"] == "v1"
    assert await oci.inspect_backend(REPO, "other") is None

    seen = []
    pulled = await oci.pull_backend(found, tmp_path / "work", lambda d, t: seen.append((d, t)))
    assert (pulled / "bin" / "tool").read_text().endswith("v1\n")
    assert seen and seen[-1][0] == seen[-1][1]  # progress reached the total


async def test_anonymous_clients_cannot_push(registry, tmp_path):
    ref = oci.backend_ref(REPO, "mock")
    async with oci.RegistryClient(ref) as reg:
        with pytest.raises(oci.OciError):
            await reg.put_manifest("mock", {"schemaVersion": 2})


# ── installer ────────────────────────────────────────────────────


@pytest.fixture
def installer(tmp_path, monkeypatch):
    from ocabra.config import settings

    monkeypatch.setattr(settings, "backend_oci_repository", REPO)
    inst = BackendInstaller(
        backends_dir=tmp_path / "backends",
        worker_pool=WorkerPool(),
        backend_registry={"mock": _InstallableMockBackend()},
        assume_fat_image=False,
    )
    inst._pip_progress_enabled = False
    return inst


async def _publish(tmp_path: Path, installer: BackendInstaller, marker: str) -> None:
    tree = _backend_tree(tmp_path / f"src-{marker}", installer.backends_dir / "mock", marker)
    await oci.publish_backend(tree, "mock", REPO, password="tok", log=lambda _m: None)


async def test_oci_install_and_update_swap(registry, installer, tmp_path):
    await installer.start()
    await _publish(tmp_path, installer, "v1")
    states = [s async for s in installer.install("mock", method="auto")]
    final = states[-1]
    assert final.install_status == BackendInstallStatus.INSTALLED, final.error
    assert final.install_source == "oci" and final.oci_digest
    target = installer.backends_dir / "mock"
    meta = json.loads((target / "metadata.json").read_text())
    assert meta["install_source"] == "oci" and meta["oci_ref"] == f"{REPO}:mock"
    assert (target / "bin" / "tool").read_text().endswith("v1\n")

    # A newer image: update swaps the folder in.
    await _publish(tmp_path, installer, "v2")
    published = await installer.published("mock", refresh=True)
    assert published["digest"] != final.oci_digest
    states = [s async for s in installer.install("mock", method="oci")]
    assert states[-1].install_status == BackendInstallStatus.INSTALLED
    assert (target / "bin" / "tool").read_text().endswith("v2\n")
    leftovers = [p.name for p in installer.backends_dir.iterdir() if p.name.startswith(".")]
    assert leftovers == []  # staging and old folder removed


async def test_image_built_elsewhere_is_refused_and_old_install_kept(registry, installer, tmp_path):
    await installer.start()
    await _publish(tmp_path, installer, "v1")
    assert [s async for s in installer.install("mock", method="oci")][-1].install_source == "oci"
    # Published from a folder at another path: its binaries point elsewhere.
    tree = _backend_tree(tmp_path / "src-bad", Path("/somewhere/else/mock"), "bad")
    await oci.publish_backend(tree, "mock", REPO, password="tok", log=lambda _m: None)
    final = [s async for s in installer.install("mock", method="oci")][-1]
    assert final.install_status == BackendInstallStatus.ERROR
    assert "outside the backend" in final.error
    target = installer.backends_dir / "mock"
    assert (target / "bin" / "tool").read_text().endswith("v1\n")  # untouched


async def test_auto_builds_from_source_when_nothing_is_published(registry, installer, monkeypatch):
    calls = []

    async def _run(self, backend_type, argv, **kw):
        calls.append(argv)
        if argv[1:3] == ["-m", "venv"]:
            (Path(argv[-1]) / "bin").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(BackendInstaller, "_run_subprocess", _run)
    await installer.start()
    final = [s async for s in installer.install("mock")][-1]  # default = auto
    assert final.install_status == BackendInstallStatus.INSTALLED
    assert final.install_source == "source"
    assert any("venv" in a for a in calls)


async def test_image_needing_a_newer_cuda_than_the_driver_is_refused(
    registry, installer, tmp_path, monkeypatch
):
    """Production's torch is a cu130 build; a pod whose driver stops at 12.8
    would get a backend that doesn't start."""
    import ocabra.core.cuda_probe as probe

    tree = _backend_tree(tmp_path / "src", installer.backends_dir / "mock")
    meta = json.loads((tree / "metadata.json").read_text())
    (tree / "metadata.json").write_text(json.dumps({**meta, "cuda": "13.0"}))
    out = await oci.publish_backend(tree, "mock", REPO, password="tok", log=lambda _m: None)
    assert out["labels"]["org.ocabra.backend.cuda"] == "13.0"

    monkeypatch.setattr(probe, "probe_cuda", lambda: {"ok": True, "driver_cuda": "12.8"})
    await installer.start()
    final = [s async for s in installer.install("mock", method="oci")][-1]
    assert final.install_status == BackendInstallStatus.ERROR
    assert "needs CUDA 13.0" in final.error and "12.8" in final.error

    monkeypatch.setattr(probe, "probe_cuda", lambda: {"ok": True, "driver_cuda": "13.1"})
    final = [s async for s in installer.install("mock", method="oci")][-1]
    assert final.install_status == BackendInstallStatus.INSTALLED, final.error
