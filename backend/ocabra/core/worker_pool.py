from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from ocabra.backends.base import BackendInterface, WorkerInfo
from ocabra.config import settings

logger = structlog.get_logger(__name__)


class WorkerPool:
    def __init__(self) -> None:
        self._backends: dict[str, BackendInterface] = {}
        self._disabled_backends: dict[str, str] = {}
        self._workers: dict[str, WorkerInfo] = {}
        self._used_ports: set[int] = set()

    def register_backend(self, backend_type: str, backend: BackendInterface) -> None:
        self._backends[backend_type] = backend
        self._disabled_backends.pop(backend_type, None)
        logger.info("backend_registered", backend_type=backend_type)

    def register_disabled_backend(self, backend_type: str, reason: str) -> None:
        self._backends.pop(backend_type, None)
        self._disabled_backends[backend_type] = reason
        logger.info("backend_disabled", backend_type=backend_type, reason=reason)

    async def get_backend(self, backend_type: str) -> BackendInterface:
        if backend_type in self._disabled_backends:
            raise RuntimeError(
                f"Backend '{backend_type}' is disabled: {self._disabled_backends[backend_type]}"
            )
        if backend_type not in self._backends:
            raise KeyError(f"Backend '{backend_type}' not registered")
        return self._backends[backend_type]

    def get_worker(self, model_id: str) -> WorkerInfo | None:
        return self._workers.get(model_id)

    def set_worker(self, model_id: str, info: WorkerInfo) -> None:
        self._workers[model_id] = info
        self._used_ports.add(info.port)

    def remove_worker(self, model_id: str) -> None:
        info = self._workers.pop(model_id, None)
        if info:
            self._used_ports.discard(info.port)

    async def assign_port(self) -> int:
        for port in range(
            settings.worker_port_range_start, settings.worker_port_range_end
        ):
            if port not in self._used_ports:
                self._used_ports.add(port)
                return port
        raise RuntimeError("No available ports in worker port range")

    def release_port(self, port: int) -> None:
        self._used_ports.discard(port)

    async def forward_request(
        self, model_id: str, path: str, body: dict
    ) -> Any:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"No worker found for model '{model_id}'")
        if worker.backend_type == "ollama":
            base = settings.ollama_base_url.rstrip("/")
            url = f"{base}{path}"
        else:
            url = f"http://127.0.0.1:{worker.port}{path}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            return resp.json()

    async def forward_stream(
        self, model_id: str, path: str, body: dict
    ) -> AsyncIterator[bytes]:
        worker = self._workers.get(model_id)
        if not worker:
            raise KeyError(f"No worker found for model '{model_id}'")
        if worker.backend_type == "ollama":
            base = settings.ollama_base_url.rstrip("/")
            url = f"{base}{path}"
        else:
            url = f"http://127.0.0.1:{worker.port}{path}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=body) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk
