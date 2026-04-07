"""MockBackend for testing — simulates load/unload without real processes."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ocabra.backends.base import BackendCapabilities, BackendInterface, ModalityType, WorkerInfo


class MockBackend(BackendInterface):

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.TEXT_GENERATION}
    def __init__(self, load_delay: float = 0.01, vram_mb: int = 4096) -> None:
        self._load_delay = load_delay
        self._vram_mb = vram_mb
        self._port_counter = 18000

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        await asyncio.sleep(self._load_delay)
        port = self._port_counter
        self._port_counter += 1
        return WorkerInfo(
            backend_type="mock",
            model_id=model_id,
            gpu_indices=gpu_indices,
            port=port,
            pid=12345,
            vram_used_mb=self._vram_mb,
        )

    async def unload(self, model_id: str) -> None:
        await asyncio.sleep(self._load_delay)

    async def health_check(self, model_id: str) -> bool:
        return True

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        return BackendCapabilities(chat=True, streaming=True, context_length=4096)

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        return self._vram_mb

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        return {"mock": True, "model_id": model_id, "path": path}

    async def forward_stream(
        self, model_id: str, path: str, body: dict
    ) -> AsyncIterator[bytes]:
        yield b"data: {}\n\n"
