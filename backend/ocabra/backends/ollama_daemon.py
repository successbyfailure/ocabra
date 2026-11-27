"""Daemon de Ollama propio en la imagen ``ocabra-node`` (pods).

En el hub Ollama es un contenedor aparte; en un pod no hay Docker ni otro
contenedor, y meter Ollama en la imagen la engordaría ~1,5 GB para todos los
pods aunque casi nunca se use. Con ``OLLAMA_NODE_AUTOSTART`` (la imagen del
nodo lo activa), la primera vez que un modelo de Ollama lo necesita se baja el
release oficial de la versión pedida (``OLLAMA_NODE_VERSION``, la del hub al
lanzar el pod, para medir lo mismo) y se arranca ``ollama serve`` contra el
mismo almacén en el que escriben las descargas (``ollama_root()``), con el
paralelismo y el contexto de los ajustes de oCabra.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import httpx
import structlog

from ocabra.config import settings

logger = structlog.get_logger(__name__)

DEFAULT_VERSION = "0.40.17"
RELEASE_URL = (
    "https://github.com/ollama/ollama/releases/download/v{version}/ollama-linux-amd64.tar.zst"
)
START_TIMEOUT_S = 90.0

_lock = asyncio.Lock()
_process: asyncio.subprocess.Process | None = None


def _version() -> str:
    return (settings.ollama_node_version or DEFAULT_VERSION).lstrip("v")


def install_dir(version: str | None = None) -> Path:
    return Path(settings.backends_dir) / "ollama" / (version or _version())


async def reachable(url: str | None = None) -> bool:
    base = (url or settings.ollama_base_url).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base}/api/version")
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def _download(version: str, target: Path) -> Path:
    """Baja y descomprime el release; devuelve el binario."""
    binary = target / "bin" / "ollama"
    if binary.is_file():
        return binary
    if shutil.which("zstd") is None:
        raise RuntimeError("Falta zstd en la imagen para descomprimir Ollama")
    staging = target.with_name(f".{target.name}.partial")
    await asyncio.to_thread(shutil.rmtree, staging, True)
    await asyncio.to_thread(staging.mkdir, parents=True, exist_ok=True)
    archive = staging / "ollama.tar.zst"
    url = RELEASE_URL.format(version=version)
    logger.info("ollama_node_download", version=version, url=url)
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=300.0)) as client:
        async with client.stream("GET", url, follow_redirects=True) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"No se pudo bajar Ollama {version}: HTTP {resp.status_code}")
            with archive.open("wb") as handle:  # noqa: ASYNC230 — chunks, no bloquea entero
                async for chunk in resp.aiter_bytes(1 << 20):
                    handle.write(chunk)
    proc = await asyncio.create_subprocess_exec(
        "tar", "--use-compress-program=zstd", "-xf", str(archive), "-C", str(staging),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"No se pudo descomprimir Ollama: {err.decode(errors='replace')[:200]}")
    archive.unlink(missing_ok=True)
    if not (staging / "bin" / "ollama").is_file():
        raise RuntimeError("El release de Ollama no trae bin/ollama")
    await asyncio.to_thread(shutil.rmtree, target, True)
    await asyncio.to_thread(staging.rename, target)
    return binary


def daemon_env() -> dict[str, str]:
    """Entorno de ``ollama serve``: el mismo que el contenedor del hub."""
    from ocabra.core.artifacts import ollama_root

    parsed = urlparse(settings.ollama_base_url)
    env = dict(os.environ)
    env.update(
        {
            "OLLAMA_HOST": f"{parsed.hostname or '127.0.0.1'}:{parsed.port or 11434}",
            "OLLAMA_MODELS": ollama_root(),
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KV_CACHE_TYPE": "q8_0",
            "OLLAMA_NUM_PARALLEL": str(int(settings.ollama_num_parallel)),
            "OLLAMA_CONTEXT_LENGTH": str(int(settings.ollama_max_context)),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
    )
    return env


async def ensure_daemon() -> bool:
    """Arranca el Ollama del nodo si hace falta. ``True`` si hay daemon.

    Sin ``OLLAMA_NODE_AUTOSTART`` no hace nada (el hub usa su contenedor).
    """
    global _process
    if not settings.ollama_node_autostart:
        return await reachable()
    if await reachable():
        return True
    async with _lock:
        if await reachable():
            return True
        version = _version()
        target = install_dir(version)
        binary = await _download(version, target)
        from ocabra.core.artifacts import ollama_root

        await asyncio.to_thread(Path(ollama_root()).mkdir, parents=True, exist_ok=True)
        log = (target / "daemon.log").open("ab")  # noqa: SIM115, ASYNC230 — vive con el proceso
        _process = await asyncio.create_subprocess_exec(
            str(binary), "serve", env=daemon_env(), stdout=log, stderr=log,
            start_new_session=True,
        )  # fmt: skip
        loop = asyncio.get_running_loop()
        deadline = loop.time() + START_TIMEOUT_S
        while loop.time() < deadline:
            if _process.returncode is not None:
                raise RuntimeError(
                    f"ollama serve terminó con código {_process.returncode} (ver {target}/daemon.log)"
                )
            if await reachable():
                logger.info("ollama_node_daemon_started", version=version, pid=_process.pid)
                return True
            await asyncio.sleep(1.0)
        raise RuntimeError("ollama serve no respondió a tiempo")


def installed_version() -> str | None:
    """Versión del Ollama propio del nodo, si está bajado."""
    if not settings.ollama_node_autostart:
        return None
    version = _version()
    return version if (install_dir(version) / "bin" / "ollama").is_file() else None
