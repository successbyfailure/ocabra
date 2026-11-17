"""Backends as OCI images: pack, push and pull without a docker daemon.

A backend is the directory ``<backends_dir>/<backend_type>`` (a venv, native
binaries, ``metadata.json``...). Published, it is an ordinary OCI image whose
layers are gzip tarballs of that tree under ``backend/``, split in chunks of
about :data:`LAYER_BYTES` so a pull can fetch them in parallel. All backends
live in **one** repository (``settings.backend_oci_repository``), one tag per
backend: a moving ``<type>`` tag and an immutable ``<type>-<date>-<hash>`` one.
One repository means one package to make public on GHCR, and nodes pull
anonymously.

Pulling is plain registry HTTP (token → manifest → blobs, sha256-checked), the
same way the Ollama registry pull works, so a node needs no docker socket.

Only the standard library and ``httpx``: the module also runs as a CLI in a
bare build container (``python -m ocabra.core.oci_registry publish ...``).
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_INDEX = "application/vnd.oci.image.index.v1+json"
OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
OCI_LAYER_GZIP = "application/vnd.oci.image.layer.v1.tar+gzip"
DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
DOCKER_LIST = "application/vnd.docker.distribution.manifest.list.v2+json"
ACCEPT = ", ".join((OCI_MANIFEST, OCI_INDEX, DOCKER_MANIFEST, DOCKER_LIST))

ROOT = "backend"  # path of the tree inside the image
LAYER_BYTES = 1 << 30
UPLOAD_CHUNK = 8 << 20
PARALLEL_BLOBS = 4
LABEL_PREFIX = "org.ocabra.backend."
REPO_SOURCE = "https://github.com/successbyfailure/ocabra"

Progress = Callable[[int, int], None]  # (bytes done, bytes total)


class OciError(RuntimeError):
    """Registry or image problem (missing tag, bad digest, wrong image...)."""


@dataclass(frozen=True)
class OciRef:
    registry: str
    repository: str
    tag: str

    @classmethod
    def parse(cls, ref: str) -> OciRef:
        ref = ref.strip()
        if "/" not in ref:
            raise OciError(f"Not a full image reference: {ref!r}")
        registry, rest = ref.split("/", 1)
        if ":" in rest.rsplit("/", 1)[-1]:
            repository, tag = rest.rsplit(":", 1)
        else:
            repository, tag = rest, "latest"
        return cls(registry, repository, tag)

    def with_tag(self, tag: str) -> OciRef:
        return OciRef(self.registry, self.repository, tag)

    def __str__(self) -> str:
        return f"{self.registry}/{self.repository}:{self.tag}"


def backend_ref(repository: str, backend_type: str, tag: str | None = None) -> OciRef:
    """``<repository>:<backend_type>`` (or another tag of that repository)."""
    return OciRef.parse(f"{repository}:{tag or backend_type}")


# ── registry client ──────────────────────────────────────────────


class RegistryClient:
    """Anonymous pulls; pushes with ``username`` + ``password`` (a GHCR token)."""

    def __init__(
        self,
        ref: OciRef,
        *,
        username: str | None = None,
        password: str | None = None,
        client: httpx.AsyncClient | None = None,
        pull_only: bool = False,
    ) -> None:
        self.ref = ref
        self._actions = "pull" if pull_only else "pull,push"
        self._auth = (username or "oauth", password) if password else None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=300.0), follow_redirects=True
        )
        self._own_client = client is None
        self._token: str | None = None
        self._base = f"https://{ref.registry}/v2/{ref.repository}"

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()

    async def __aenter__(self) -> RegistryClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _authenticate(self, challenge: str) -> None:
        params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
        realm = params.pop("realm", None)
        if not realm:
            raise OciError(f"Registry asked for unsupported auth: {challenge}")
        if self._auth:
            params["scope"] = f"repository:{self.ref.repository}:{self._actions}"
        resp = await self._client.get(realm, params=params, auth=self._auth)
        if resp.status_code != 200:
            raise OciError(f"Registry token refused ({resp.status_code}): {resp.text[:200]}")
        body = resp.json()
        self._token = body.get("token") or body.get("access_token")

    async def _request(self, method: str, url: str, **kw: Any) -> httpx.Response:
        for _attempt in range(2):
            headers = dict(kw.pop("headers", None) or {})
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            resp = await self._client.request(method, url, headers=headers, **kw)
            if resp.status_code != 401:
                return resp
            await self._authenticate(resp.headers.get("www-authenticate", ""))
            kw["headers"] = {k: v for k, v in headers.items() if k != "Authorization"}
        return resp

    async def manifest(self, tag: str | None = None) -> tuple[dict[str, Any], str]:
        """``(manifest, digest)`` of *tag*; a multi-arch index resolves to linux/amd64."""
        ref = tag or self.ref.tag
        resp = await self._request(
            "GET", f"{self._base}/manifests/{ref}", headers={"Accept": ACCEPT}
        )
        if resp.status_code == 404:
            raise OciError(f"{self.ref.with_tag(ref)} is not published")
        if resp.status_code >= 400:
            raise OciError(
                f"Manifest of {self.ref.with_tag(ref)}: {resp.status_code} {resp.text[:200]}"
            )
        body = resp.json()
        digest = resp.headers.get("docker-content-digest") or _sha256(resp.content)
        if body.get("mediaType") in (OCI_INDEX, DOCKER_LIST) or "manifests" in body:
            for entry in body.get("manifests") or []:
                plat = entry.get("platform") or {}
                if (
                    plat.get("os", "linux") == "linux"
                    and plat.get("architecture", "amd64") == "amd64"
                ):
                    return await self.manifest(entry["digest"])
            raise OciError(f"{self.ref.with_tag(ref)} has no linux/amd64 image")
        return body, digest

    async def digest(self, tag: str | None = None) -> str | None:
        """Digest *tag* points to, or ``None`` when it isn't published."""
        ref = tag or self.ref.tag
        resp = await self._request(
            "HEAD", f"{self._base}/manifests/{ref}", headers={"Accept": ACCEPT}
        )
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise OciError(f"{self.ref.with_tag(ref)}: {resp.status_code}")
        return resp.headers.get("docker-content-digest")

    async def tags(self) -> list[str]:
        """Every tag of the repository (follows ``Link`` pagination)."""
        out: list[str] = []
        url: str | None = f"{self._base}/tags/list?n=1000"
        while url:
            resp = await self._request("GET", url)
            if resp.status_code == 404:
                return out
            if resp.status_code >= 400:
                raise OciError(f"Tags of {self.ref.repository}: {resp.status_code}")
            out.extend(resp.json().get("tags") or [])
            link = resp.headers.get("link", "")
            match = re.search(r"<([^>]+)>", link)
            url = (
                (
                    match.group(1)
                    if match.group(1).startswith("http")
                    else f"https://{self.ref.registry}{match.group(1)}"
                )
                if match
                else None
            )
        return out

    async def blob_json(self, digest: str) -> dict[str, Any]:
        resp = await self._request("GET", f"{self._base}/blobs/{digest}")
        if resp.status_code >= 400:
            raise OciError(f"Blob {digest}: {resp.status_code}")
        if _sha256(resp.content) != digest:
            raise OciError(f"Blob {digest} does not match its digest")
        return resp.json()

    async def download_blob(
        self, digest: str, dest: Path, on_bytes: Callable[[int], None] | None = None
    ) -> None:
        """Stream blob *digest* into *dest*, verifying its sha256."""
        h = hashlib.sha256()
        url = f"{self._base}/blobs/{digest}"
        for _attempt in range(2):
            headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
            async with self._client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 401:
                    await self._authenticate(resp.headers.get("www-authenticate", ""))
                    continue
                if resp.status_code >= 400:
                    raise OciError(f"Blob {digest}: {resp.status_code}")
                with dest.open("wb") as fh:  # noqa: ASYNC230 — sequential writes of a stream
                    async for chunk in resp.aiter_bytes(1 << 20):
                        fh.write(chunk)
                        h.update(chunk)
                        if on_bytes:
                            on_bytes(len(chunk))
            break
        if f"sha256:{h.hexdigest()}" != digest:
            await asyncio.to_thread(dest.unlink, True)
            raise OciError(f"Blob {digest} arrived corrupted (sha256 mismatch)")

    async def blob_exists(self, digest: str) -> bool:
        resp = await self._request("HEAD", f"{self._base}/blobs/{digest}")
        return resp.status_code == 200

    async def upload_blob(
        self,
        path: Path | None,
        digest: str,
        *,
        data: bytes | None = None,
        on_bytes: Callable[[int], None] | None = None,
    ) -> None:
        """Upload a blob (chunked, so multi-GB layers work) unless it is there already."""
        if await self.blob_exists(digest):
            if on_bytes and path is not None:
                on_bytes(await asyncio.to_thread(os.path.getsize, path))
            return
        resp = await self._request("POST", f"{self._base}/blobs/uploads/")
        if resp.status_code != 202:
            raise OciError(f"Upload refused ({resp.status_code}): {resp.text[:200]}")
        location = _absolute(resp.headers["location"], self.ref.registry)
        # One streamed PATCH, as docker does: registries cap ranged chunks
        # (GHCR refuses Content-Range chunks over 4 MiB) but not a stream.
        if data is not None:
            content: Any = data
        else:

            async def _stream():
                with path.open("rb") as fh:  # type: ignore[union-attr]  # noqa: ASYNC230
                    while True:
                        chunk = await asyncio.to_thread(fh.read, UPLOAD_CHUNK)
                        if not chunk:
                            break
                        if on_bytes:
                            on_bytes(len(chunk))
                        yield chunk

            content = _stream()
        resp = await self._request(
            "PATCH",
            location,
            content=content,
            headers={"Content-Type": "application/octet-stream"},
        )
        if resp.status_code != 202:
            raise OciError(f"Upload refused ({resp.status_code}): {resp.text[:200]}")
        location = _absolute(resp.headers.get("location", location), self.ref.registry)
        sep = "&" if "?" in location else "?"
        resp = await self._request("PUT", f"{location}{sep}digest={digest}")
        if resp.status_code != 201:
            raise OciError(f"Upload not accepted ({resp.status_code}): {resp.text[:200]}")

    async def put_manifest(self, tag: str, manifest: dict[str, Any]) -> str:
        body = _canonical(manifest)
        resp = await self._request(
            "PUT",
            f"{self._base}/manifests/{tag}",
            content=body,
            headers={"Content-Type": manifest.get("mediaType", OCI_MANIFEST)},
        )
        if resp.status_code not in (200, 201):
            raise OciError(f"Manifest {tag} refused ({resp.status_code}): {resp.text[:200]}")
        return _sha256(body)


# ── packing / unpacking ──────────────────────────────────────────


@dataclass
class Layer:
    path: Path
    digest: str  # sha256 of the gzip blob
    diff_id: str  # sha256 of the uncompressed tar
    size: int


class _HashingWriter(io.RawIOBase):
    """File-like that writes through and hashes what passes."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw
        self.h = hashlib.sha256()
        self.n = 0

    def writable(self) -> bool:
        return True

    def write(self, b: Any) -> int:
        self.h.update(b)
        self.n += len(b)
        self.raw.write(b)
        return len(b)


def _walk(root: Path) -> tuple[list[Path], list[Path]]:
    """Directories and files under *root*, skipping mount points inside it.

    A mount point is something bind-mounted into the backend's folder (e.g.
    ACE-Step's checkpoints): host data, not part of the backend.
    """
    dirs: list[Path] = []
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in sorted(dirnames):
            full = Path(dirpath, d)
            if os.path.ismount(full):
                continue
            if full.is_symlink():
                files.append(full)
            else:
                keep.append(d)
                dirs.append(full)
        dirnames[:] = keep
        files.extend(Path(dirpath, f) for f in sorted(filenames))
    return dirs, files


def _normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Same bytes for the same tree: no owners, no access/change times."""
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.pax_headers = {}
    return info


def pack_tree(root: Path, out_dir: Path, layer_bytes: int = LAYER_BYTES) -> list[Layer]:
    """Tar+gzip *root* into layers of about *layer_bytes* (uncompressed).

    Every file is stored whole in exactly one layer (hard links are written as
    plain files: a link pointing into another layer would not resolve).
    Directories all go in the first layer. Deterministic: packing an
    unchanged tree again gives the same digests, so republishing it uploads
    nothing (the registry already has those blobs).
    """
    root = root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs, files = _walk(root)
    groups: list[list[Path]] = [[]]
    size = 0
    for f in files:
        fsize = 0 if f.is_symlink() else f.stat().st_size
        if groups[-1] and size + fsize > layer_bytes:
            groups.append([])
            size = 0
        groups[-1].append(f)
        size += fsize
    layers: list[Layer] = []
    for i, group in enumerate(groups):
        path = out_dir / f"layer-{i:03d}.tar.gz"
        with path.open("wb") as raw:
            compressed = _HashingWriter(raw)
            with gzip.GzipFile(fileobj=compressed, mode="wb", compresslevel=3, mtime=0) as gz:
                tar_hash = _HashingWriter(gz)
                with tarfile.open(fileobj=tar_hash, mode="w|", format=tarfile.GNU_FORMAT) as tar:
                    if i == 0:
                        tar.add(root, arcname=ROOT, recursive=False, filter=_normalize)
                        for d in dirs:
                            tar.add(
                                d,
                                arcname=f"{ROOT}/{d.relative_to(root)}",
                                recursive=False,
                                filter=_normalize,
                            )
                    for f in group:
                        tar.inodes.clear()  # no cross-file hard links
                        tar.add(
                            f,
                            arcname=f"{ROOT}/{f.relative_to(root)}",
                            recursive=False,
                            filter=_normalize,
                        )
        layers.append(
            Layer(
                path=path,
                digest=f"sha256:{compressed.h.hexdigest()}",
                diff_id=f"sha256:{tar_hash.h.hexdigest()}",
                size=compressed.n,
            )
        )
    return layers


def unpack_layers(layer_paths: list[Path], dest: Path) -> Path:
    """Extract layers in order into *dest*; returns the backend tree inside it.

    Uses tarfile's ``tar`` filter: no member may land outside *dest* (absolute
    or ``..`` names are refused) while symlinks are kept as they are — a venv's
    ``bin/python`` points to the system interpreter by absolute path.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for path in layer_paths:
        with tarfile.open(path, mode="r:gz") as tar:
            for member in tar:
                name = member.name.lstrip("./")
                base = os.path.basename(name)
                if base.startswith(".wh."):  # OCI whiteout: delete from lower layers
                    target = dest / os.path.dirname(name) / base[4:]
                    if target.is_dir() and not target.is_symlink():
                        shutil.rmtree(target, ignore_errors=True)
                    else:
                        target.unlink(missing_ok=True)
                    continue
                tar.extract(member, dest, filter="tar")
    tree = dest / ROOT
    if not tree.is_dir():
        raise OciError(f"The image has no '{ROOT}/' tree: not an oCabra backend image")
    return tree


# ── backend metadata carried in the image ────────────────────────


_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def package_names(requirements: list[str]) -> list[str]:
    """Distribution names out of pip requirement strings."""
    out = []
    for req in requirements:
        m = _NAME_RE.match(req)
        if m:
            out.append(m.group(1).lower().replace("_", "-"))
    return out


def installed_packages(python_bin: str | None, names: list[str]) -> dict[str, str]:
    """``{name: version}`` (``version+git.<sha>`` for VCS installs) in a venv."""
    if not python_bin:
        return {}
    venv = Path(python_bin).parent.parent
    sites = sorted(venv.glob("lib/python*/site-packages"))
    if not sites:
        return {}
    wanted = {n.lower().replace("_", "-") for n in names}
    out: dict[str, str] = {}
    for info in sites[0].glob("*.dist-info"):
        stem = info.name[: -len(".dist-info")]
        name, _, version = stem.rpartition("-")
        norm = name.lower().replace("_", "-")
        if norm not in wanted:
            continue
        direct = info / "direct_url.json"
        if direct.is_file():
            try:
                commit = json.loads(direct.read_text()).get("vcs_info", {}).get("commit_id")
                if commit:
                    version = f"{version}+git.{commit[:10]}"
            except (OSError, ValueError):
                pass
        out[norm] = version
    return dict(sorted(out.items()))


def describe_tree(tree: Path) -> dict[str, Any]:
    """What a backend folder holds: type, version, key package versions."""
    meta_path = tree / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    python_bin = meta.get("python_bin")
    packages = installed_packages(
        python_bin if python_bin and Path(python_bin).exists() else None,
        package_names(meta.get("pip_packages") or []),
    )
    python = None
    cfg = Path(python_bin).parent.parent / "pyvenv.cfg" if python_bin else None
    if cfg is not None and cfg.is_file():
        m = re.search(r"^version(?:_info)?\s*=\s*(\d+\.\d+)", cfg.read_text(), re.M)
        python = m.group(1) if m else None
    return {
        "metadata": meta,
        "version": _headline_version(meta, packages),
        "packages": packages,
        "python": python,
        "cuda": _required_cuda(tree, meta, python_bin),
    }


def _headline_version(meta: dict[str, Any], packages: dict[str, str]) -> str:
    """One readable version: the backend's own package, else the first one that
    isn't the generic stack; native builds say their commit."""
    backend = str(meta.get("backend_type") or "").replace("_", "-")
    generic = {"torch", "torchvision", "numpy", "pillow", "fastapi", "uvicorn"}
    for exact in (True, False):
        for name, version in packages.items():
            if (name == backend) if exact else (backend in name or name in backend):
                return f"{name} {version}"
    for req in package_names(meta.get("pip_packages") or []):
        if req not in generic and req in packages:
            return f"{req} {packages[req]}"
    version = str(meta.get("version") or "")
    if version and version != "source" and not any(op in version for op in "<>=@"):
        return version
    if meta.get("git_ref"):
        return str(meta["git_ref"])[:12]
    if packages:
        name, ver = next(iter(packages.items()))
        return f"{name} {ver}"
    return version


def _required_cuda(tree: Path, meta: dict[str, Any], python_bin: str | None) -> str | None:
    """CUDA the backend was built against: torch's build, or the native build's."""
    if meta.get("cuda"):
        return str(meta["cuda"])
    if python_bin:
        venv = Path(python_bin).parent.parent
        for version_py in venv.glob("lib/python*/site-packages/torch/version.py"):
            m = re.search(r"^cuda[^=]*=\s*'([\d.]+)'", version_py.read_text(), re.M)
            if m:
                return m.group(1)
    return None


def cuda_tuple(version: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", version or "")[:2])


# ── publish / pull ───────────────────────────────────────────────


async def publish_backend(
    tree: Path,
    backend_type: str,
    repository: str,
    *,
    password: str,
    username: str | None = None,
    work_dir: Path | None = None,
    gpu_archs: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Pack *tree* and push it as ``<repository>:<backend_type>`` (+ an immutable tag)."""
    info = describe_tree(tree)
    meta = info["metadata"]
    if meta.get("backend_type") not in (None, backend_type):
        raise OciError(f"{tree} holds backend {meta.get('backend_type')!r}, not {backend_type!r}")
    # Own folder per run: two publishes of the same backend (a retried job)
    # must not delete each other's layers.
    work = (
        work_dir or tree.parent / f".oci-publish-{backend_type}-{os.getpid()}-{os.urandom(3).hex()}"
    )
    shutil.rmtree(work, ignore_errors=True)
    try:
        log(f"packing {tree} …")
        layers = await asyncio.to_thread(pack_tree, tree, work)
        total = sum(layer.size for layer in layers)
        log(f"{len(layers)} layers, {total / 1e9:.2f} GB compressed")
        version_hash = hashlib.sha256("".join(lyr.diff_id for lyr in layers).encode()).hexdigest()[
            :10
        ]
        stamp = datetime.now(UTC)
        labels = {
            f"{LABEL_PREFIX}type": backend_type,
            f"{LABEL_PREFIX}version": info["version"],
            f"{LABEL_PREFIX}packages": json.dumps(info["packages"], sort_keys=True),
            f"{LABEL_PREFIX}git-ref": str(meta.get("git_ref") or ""),
            f"{LABEL_PREFIX}python": info["python"] or "",
            f"{LABEL_PREFIX}gpu-archs": gpu_archs or "",
            f"{LABEL_PREFIX}cuda": info["cuda"] or "",
            f"{LABEL_PREFIX}content": version_hash,
            "org.opencontainers.image.source": REPO_SOURCE,
            "org.opencontainers.image.created": stamp.isoformat(),
            "org.opencontainers.image.title": f"ocabra backend {backend_type}",
        }
        config = {
            "architecture": "amd64",
            "os": "linux",
            "created": stamp.isoformat(),
            "config": {"Labels": labels},
            "rootfs": {"type": "layers", "diff_ids": [lyr.diff_id for lyr in layers]},
        }
        config_bytes = _canonical(config)
        config_digest = _sha256(config_bytes)
        ref = backend_ref(repository, backend_type)
        done = 0

        def _bytes(n: int) -> None:
            nonlocal done
            done += n

        async with RegistryClient(ref, username=username, password=password) as reg:
            await reg.upload_blob(None, config_digest, data=config_bytes)
            for i, layer in enumerate(layers):
                log(f"uploading layer {i + 1}/{len(layers)} ({layer.size / 1e9:.2f} GB)")
                await reg.upload_blob(layer.path, layer.digest, on_bytes=_bytes)
            manifest = {
                "schemaVersion": 2,
                "mediaType": OCI_MANIFEST,
                "config": {
                    "mediaType": OCI_CONFIG,
                    "digest": config_digest,
                    "size": len(config_bytes),
                },
                "layers": [
                    {"mediaType": OCI_LAYER_GZIP, "digest": lyr.digest, "size": lyr.size}
                    for lyr in layers
                ],
                "annotations": {
                    k: v for k, v in labels.items() if k.startswith("org.opencontainers")
                },
            }
            immutable = f"{backend_type}-{stamp:%Y%m%d}-{version_hash}"
            digest = await reg.put_manifest(immutable, manifest)
            await reg.put_manifest(backend_type, manifest)
        log(f"published {ref.with_tag(immutable)} = {ref} ({digest})")
        return {
            "ref": str(ref),
            "immutable": str(ref.with_tag(immutable)),
            "digest": digest,
            "labels": labels,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


@dataclass
class PublishedBackend:
    ref: OciRef
    digest: str
    labels: dict[str, str]
    layers: list[dict[str, Any]]

    @property
    def size(self) -> int:
        return sum(int(lyr.get("size") or 0) for lyr in self.layers)

    @property
    def packages(self) -> dict[str, str]:
        try:
            return json.loads(self.labels.get(f"{LABEL_PREFIX}packages") or "{}")
        except ValueError:
            return {}

    def summary(self) -> dict[str, Any]:
        return {
            "ref": str(self.ref),
            "digest": self.digest,
            "size_bytes": self.size,
            "version": self.labels.get(f"{LABEL_PREFIX}version"),
            "packages": self.packages,
            "gpu_archs": self.labels.get(f"{LABEL_PREFIX}gpu-archs") or None,
            "cuda": self.labels.get(f"{LABEL_PREFIX}cuda") or None,
            "created": self.labels.get("org.opencontainers.image.created"),
        }


def _pull_password() -> str | None:
    """Token for a private repository (``BACKEND_OCI_TOKEN``); public ones need none."""
    return os.environ.get("BACKEND_OCI_TOKEN") or None


async def inspect_backend(repository: str, backend_type: str) -> PublishedBackend | None:
    """What is published for *backend_type*, or ``None`` if nothing is."""
    ref = backend_ref(repository, backend_type)
    async with RegistryClient(ref, password=_pull_password(), pull_only=True) as reg:
        try:
            manifest, digest = await reg.manifest()
        except OciError as exc:
            if "is not published" in str(exc):
                return None
            raise
        config = await reg.blob_json(manifest["config"]["digest"])
    labels = (config.get("config") or {}).get("Labels") or {}
    if labels.get(f"{LABEL_PREFIX}type") != backend_type:
        raise OciError(f"{ref} is not an oCabra backend image for {backend_type}")
    return PublishedBackend(ref, digest, labels, manifest.get("layers") or [])


async def pull_backend(
    published: PublishedBackend, work_dir: Path, progress: Progress | None = None
) -> Path:
    """Download and unpack *published* under *work_dir*; returns the tree."""
    blobs = work_dir / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    total = published.size or 1
    done = 0

    def _bytes(n: int) -> None:
        nonlocal done
        done += n
        if progress:
            progress(done, total)

    sem = asyncio.Semaphore(PARALLEL_BLOBS)
    paths: list[Path] = []
    async with RegistryClient(published.ref, password=_pull_password(), pull_only=True) as reg:
        await reg.digest()  # authenticate once before the parallel downloads

        async def _one(i: int, layer: dict[str, Any]) -> None:
            async with sem:
                await reg.download_blob(layer["digest"], paths[i], _bytes)

        for i, _layer in enumerate(published.layers):
            paths.append(blobs / f"layer-{i:03d}.tar.gz")
        await asyncio.gather(*(_one(i, lyr) for i, lyr in enumerate(published.layers)))
    tree = await asyncio.to_thread(unpack_layers, paths, work_dir / "root")
    shutil.rmtree(blobs, ignore_errors=True)
    return tree


# ── helpers ──────────────────────────────────────────────────────


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _absolute(location: str, registry: str) -> str:
    return location if location.startswith("http") else f"https://{registry}{location}"


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m ocabra.core.oci_registry")
    sub = parser.add_subparsers(dest="cmd", required=True)
    pub = sub.add_parser("publish", help="pack a backend folder and push it")
    pub.add_argument("--dir", required=True, type=Path, help="the backend folder")
    pub.add_argument("--type", required=True, help="backend type, e.g. diffusers")
    pub.add_argument(
        "--repository",
        default=os.environ.get(
            "BACKEND_OCI_REPOSITORY", "ghcr.io/successbyfailure/ocabra-backends"
        ),
    )
    pub.add_argument("--username", default=os.environ.get("GHCR_USER"))
    pub.add_argument("--gpu-archs", default=None, help="CUDA archs native code was built for")
    show = sub.add_parser("inspect", help="show what is published for a backend")
    show.add_argument("--type", required=True)
    show.add_argument(
        "--repository",
        default=os.environ.get(
            "BACKEND_OCI_REPOSITORY", "ghcr.io/successbyfailure/ocabra-backends"
        ),
    )
    args = parser.parse_args(argv)
    if args.cmd == "publish":
        token = os.environ.get("GHCR_TOKEN")
        if not token:
            parser.error("set GHCR_TOKEN (e.g. GHCR_TOKEN=$(gh auth token))")
        out = asyncio.run(
            publish_backend(
                args.dir,
                args.type,
                args.repository,
                password=token,
                username=args.username,
                gpu_archs=args.gpu_archs,
            )
        )
        print(json.dumps({k: v for k, v in out.items() if k != "labels"}, indent=2))
    else:
        found = asyncio.run(inspect_backend(args.repository, args.type))
        print(json.dumps(found.summary() if found else None, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
