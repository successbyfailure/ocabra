from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


@dataclass
class BackendInstallSpec:
    """Declarative installation spec for a modular backend.

    Each backend that wants to be installable/uninstallable at runtime exposes an
    instance of this class through :pyattr:`BackendInterface.install_spec`.

    Attributes:
        oci_image: Fully-qualified OCI image used for pre-built distribution.
        oci_tags: Mapping of hardware variant → tag (e.g. ``{"cuda12": "latest-cuda12"}``).
        oci_extract_path: Path inside the OCI image that contains the backend tree.
        pip_packages: Packages to install when using the ``source`` install method.
        pip_extra_index_urls: Extra ``--extra-index-url`` values passed to pip (e.g.
            ``https://download.pytorch.org/whl/cu124`` to pull CUDA-enabled torch).
        include_core_runtime: If true, the installer seeds the venv with the core
            FastAPI/Pydantic/httpx stack that the oCabra worker scripts import.
            Backends that don't run a FastAPI worker can set this to ``False``.
        post_install_script: Optional script path (relative to repo) to run after install.
        estimated_size_mb: Approximate on-disk footprint reported to the UI.
        display_name: Human-friendly name shown on the UI.
        description: Short description of the backend.
        tags: Labels for filtering on the UI (e.g. ``["LLM", "GPU", "CUDA"]``).
        python_version: Python version required for ``source`` installs.
    """

    oci_image: str
    oci_tags: dict[str, str] = field(default_factory=dict)
    oci_extract_path: str = "/backend"
    pip_packages: list[str] = field(default_factory=list)
    pip_extra_index_urls: list[str] = field(default_factory=list)
    include_core_runtime: bool = True
    post_install_script: str | None = None
    estimated_size_mb: int = 0
    display_name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    python_version: str = "3.11"


class ModalityType(StrEnum):
    TEXT_GENERATION = "text_generation"
    EMBEDDINGS = "embeddings"
    IMAGE_GENERATION = "image_generation"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_SPEECH = "audio_speech"
    RERANKING = "reranking"


@dataclass
class BackendCapabilities:
    chat: bool = False
    completion: bool = False
    tools: bool = False
    vision: bool = False
    embeddings: bool = False
    pooling: bool = False
    rerank: bool = False
    classification: bool = False
    score: bool = False
    reasoning: bool = False
    image_generation: bool = False
    audio_transcription: bool = False
    tts: bool = False
    music_generation: bool = False
    streaming: bool = False
    context_length: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkerInfo:
    backend_type: str
    model_id: str
    gpu_indices: list[int]
    port: int
    pid: int
    vram_used_mb: int


class BackendInterface(ABC):
    @property
    def install_spec(self) -> BackendInstallSpec | None:
        """Return the installation spec for this backend, or ``None``.

        Returning ``None`` means the backend is pre-installed in the image, is an
        external service (e.g. Ollama), or otherwise does not participate in the
        modular install/uninstall flow.
        """
        return None

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

    # ------------------------------------------------------------------
    # Multi-modal interface (optional — subclasses override as needed)
    # ------------------------------------------------------------------

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        """Return the set of modalities this backend can handle.

        Subclasses MUST override this to declare their capabilities.
        """
        return set()

    async def generate_text(
        self, model_id: str, messages: list[dict], **kwargs: Any
    ) -> dict:
        """Generate a text completion. Override in text-capable backends."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support text generation"
        )

    async def stream_text(
        self, model_id: str, messages: list[dict], **kwargs: Any
    ) -> AsyncIterator[bytes]:
        """Stream a text completion. Override in text-capable backends."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support text streaming"
        )

    async def generate_embeddings(
        self, model_id: str, input: list[str], **kwargs: Any
    ) -> dict:
        """Generate embeddings for the given input texts."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support embeddings"
        )

    async def generate_image(
        self, model_id: str, prompt: str, **kwargs: Any
    ) -> dict:
        """Generate an image from a text prompt."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support image generation"
        )

    async def transcribe(
        self, model_id: str, audio: bytes, **kwargs: Any
    ) -> dict:
        """Transcribe audio to text."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support audio transcription"
        )

    async def synthesize_speech(
        self, model_id: str, text: str, **kwargs: Any
    ) -> bytes:
        """Synthesize speech from text. Returns raw audio bytes."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support speech synthesis"
        )

    async def rerank(
        self, model_id: str, query: str, documents: list[str], **kwargs: Any
    ) -> dict:
        """Rerank documents by relevance to the query."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support reranking"
        )
