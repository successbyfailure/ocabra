from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class BackendCapabilities:
    chat: bool = False
    completion: bool = False
    tools: bool = False
    vision: bool = False
    embeddings: bool = False
    reasoning: bool = False
    image_generation: bool = False
    audio_transcription: bool = False
    tts: bool = False
    streaming: bool = False
    context_length: int = 0

    def to_dict(self) -> dict:
        return {
            "chat": self.chat,
            "completion": self.completion,
            "tools": self.tools,
            "vision": self.vision,
            "embeddings": self.embeddings,
            "reasoning": self.reasoning,
            "image_generation": self.image_generation,
            "audio_transcription": self.audio_transcription,
            "tts": self.tts,
            "streaming": self.streaming,
            "context_length": self.context_length,
        }


@dataclass
class WorkerInfo:
    backend_type: str
    model_id: str
    gpu_indices: list[int]
    port: int
    pid: int
    vram_used_mb: int


class BackendInterface(ABC):
    @abstractmethod
    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        """Load the model. Returns WorkerInfo with the running process."""

    @abstractmethod
    async def unload(self, model_id: str) -> None:
        """Unload the model and free resources."""

    @abstractmethod
    async def health_check(self, model_id: str) -> bool:
        """Returns True if the worker is ready to serve requests."""

    @abstractmethod
    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        """Detect and return the model capabilities."""

    @abstractmethod
    async def get_vram_estimate_mb(self, model_id: str) -> int:
        """Estimate VRAM required before loading the model."""

    @abstractmethod
    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        """Forward a request to the worker. Returns the response."""

    @abstractmethod
    async def forward_stream(
        self, model_id: str, path: str, body: dict
    ) -> AsyncIterator[bytes]:
        """Forward a streaming request to the worker."""
