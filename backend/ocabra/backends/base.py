import socket
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class BackendInstallSpec:
    """Declarative installation spec for a modular backend.

    Each backend that wants to be installable/uninstallable at runtime exposes an
    instance of this class through :pyattr:`BackendInterface.install_spec`.

    Attributes:
        Prebuilt (OCI) distribution needs nothing here: every backend is
        published as ``<settings.backend_oci_repository>:<backend_type>`` by
        ``core/oci_registry.py`` from an installed copy.
        pip_packages: Packages to install when using the ``source`` install method.
        pip_extra_index_urls: Extra ``--extra-index-url`` values passed to pip (e.g.
            ``https://download.pytorch.org/whl/cu124`` to pull CUDA-enabled torch).
        include_core_runtime: If true, the installer seeds the venv with the core
            FastAPI/Pydantic/httpx stack that the oCabra worker scripts import.
            Backends that don't run a FastAPI worker can set this to ``False``.
        apt_packages: Debian/Ubuntu packages to ``apt-get install`` before the
            pip / git steps. Required for native backends (build-essential, cmake,
            git, etc.). The installer is best-effort: if ``apt-get`` is not
            available it logs a warning and continues — the user is then on the
            hook for providing the deps in the base image. Packages persist
            inside the running container only; they are re-installed on every
            ``install()`` (idempotent).
        git_repo: HTTPS URL of an upstream repo to clone into
            ``<backends_dir>/<name>/src/`` before running the post-install
            script. Used by ACE-Step (uv sync against project) and by native
            builds (llama.cpp, BitNet).
        git_ref: Git ref (branch, tag or commit) to check out.
        git_recursive: If true, clone with ``--recursive`` (submodules). BitNet
            requires this.
        post_install_script: Path to a shell script executed after pip install.
            May be absolute or relative to the repo root. Run with bash, cwd =
            ``<backends_dir>/<name>/``, env extended with ``BACKEND_DIR``,
            ``VENV_DIR`` (may be empty), ``PYTHON_BIN`` (may be empty),
            ``SRC_DIR`` (may be empty), ``BIN_DIR``. Used by native backends
            for cmake builds and by ACE-Step for ``uv sync``.
        extra_bins: Mapping of logical name → relative path under
            ``<backends_dir>/<name>/`` produced by the install. After the
            post-install script runs, the installer asserts each file exists,
            chmod +x's it, and persists the absolute paths in ``metadata.json``
            so the backend can resolve them at load time.
        estimated_size_mb: Approximate on-disk footprint reported to the UI.
        display_name: Human-friendly name shown on the UI.
        description: Short description of the backend.
        tags: Labels for filtering on the UI (e.g. ``["LLM", "GPU", "CUDA"]``).
        python_version: Python version required for ``source`` installs.
    """

    pip_packages: list[str] = field(default_factory=list)
    pip_extra_index_urls: list[str] = field(default_factory=list)
    include_core_runtime: bool = True
    apt_packages: list[str] = field(default_factory=list)
    git_repo: str | None = None
    git_ref: str = "main"
    git_recursive: bool = False
    post_install_script: str | None = None
    extra_bins: dict[str, str] = field(default_factory=dict)
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
    VIDEO_UPSCALING = "video_upscaling"
    VIDEO_GENERATION = "video_generation"


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
    reasoning: bool = False  # Model emits ``reasoning_content`` / chain-of-thought
    image_generation: bool = False
    # ``image_editing`` is True when the loaded pipeline has an img2img (or
    # inpainting) companion — i.e. ``/v1/images/edits`` will actually work.
    # False for text-to-image-only pipelines like QwenImage21Pipeline or the
    # distilled Z-Image-Turbo / FLUX.2 Klein variants.
    image_editing: bool = False
    audio_transcription: bool = False  # Dedicated STT (e.g. Whisper) on /v1/audio/transcriptions
    tts: bool = False
    music_generation: bool = False
    # Multimodal input in /v1/chat/completions:
    # ``audio_input``  → accepts ``{type:"input_audio", input_audio:{data, format}}`` content parts.
    # ``video_input``  → accepts ``{type:"video_url",  video_url:{url}}`` content parts.
    # ``audio_output`` → emits audio chunks in the response stream (e.g. Qwen-Omni, GPT-4o-realtime).
    # These are about *consuming/producing* media inside chat, distinct from
    # ``audio_transcription`` (a dedicated STT endpoint), ``tts`` (a dedicated
    # synthesis endpoint), or ``vision`` (image input).
    audio_input: bool = False
    video_input: bool = False
    audio_output: bool = False
    # Dedicated video super-resolution endpoint (/ocabra/video/upscale).
    # Distinct from ``video_input``, which is about consuming video *inside*
    # chat: a model can restore video without being able to talk about it.
    video_upscaling: bool = False
    # Single-image super-resolution (/v1/images/upscale). Separate from
    # ``video_upscaling`` because not every video model can take a still:
    # FlashVSR is temporal and only has a video path.
    image_upscaling: bool = False
    # Text-to-video generation (/v1/videos), e.g. LTX-2.5 through ComfyUI.
    video_generation: bool = False
    streaming: bool = False
    context_length: int = 0
    # Native context length the model was trained/configured for, independent
    # of runtime caps (OLLAMA_CONTEXT_LENGTH, use_case overrides, ...). When
    # the backend can tell them apart, ``context_length`` is the effective
    # window it will actually serve and ``native_context_length`` is the hard
    # ceiling that bounds presets. Zero means "unknown — treat context_length
    # as both".
    native_context_length: int = 0
    # Independent hard cap on output tokens per request. Zero means unknown;
    # it must never be inferred from context_length. A profile may declare an
    # enforceable operational cap through runtime_policy.max_output_tokens.
    max_output_tokens: int = 0

    # Concurrencia y forma de la cache, que hasta ahora no se publicaban y
    # cada panel tenia que adivinar. Con llama.cpp de serie la ventana se
    # reparte entre los slots (cada uno recibe ctx/parallel); con el fork de
    # Makespace y ``--kv-unified`` es un pool compartido y UNA sola peticion
    # puede usarlo entero. Sin esta distincion, un cliente escrito cuando solo
    # existia llama.cpp de serie divide por los slots y ensena un contexto que
    # no es el que se sirve.
    parallel_slots: int = 0  # 0 = desconocido
    kv_cache_shared: bool = False  # True = pool unico para todos los slots
    # Lo que de verdad puede gastar una peticion: la ventana entera cuando la
    # cache es compartida, su parte cuando no. Es el numero que quiere ensenar
    # una interfaz, y evita que lo calcule mal cada una por su cuenta.
    context_per_request: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BackendCapabilities":
        """Rebuild from a ``to_dict`` payload, ignoring unknown keys."""
        if not isinstance(data, dict):
            return cls()
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class WorkerInfo:
    backend_type: str
    model_id: str
    gpu_indices: list[int]
    port: int
    pid: int
    vram_used_mb: int
    # External backends may live at different per-GPU pool endpoints.
    endpoint_url: str | None = None


def _ephemeral_port() -> int:
    """Port picked by the kernel from the ephemeral range (fallback only)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
    async def unload(self, model_id: str, *, endpoint_url: str | None = None) -> None:
        """Unload the model and free resources.

        ``endpoint_url`` is part of the contract only for backends that serve
        the same model from several pool endpoints (Ollama): the caller threads
        the worker's endpoint so the right pool is told to drop it. Every other
        backend ignores it — one model lives in one local process.
        """

    @abstractmethod
    async def health_check(self, model_id: str) -> bool:
        """Returns True if the worker is ready to serve requests."""

    @abstractmethod
    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        """Detect and return the model capabilities."""

    @abstractmethod
    async def get_vram_estimate_mb(self, model_id: str, extra_config: dict | None = None) -> int:
        """Estimate VRAM required before loading the model.

        ``extra_config`` lets backends compute a realistic estimate for
        quantized loading paths (e.g. GGUF transformer + NF4 text encoder)
        whose on-disk safetensors size massively over-estimates the actual
        VRAM footprint. Callers should pass ``ModelState.extra_config`` when
        available; None is fine and backends must degrade gracefully.
        """

    async def prepare_request(self, model_id: str, path: str, body: dict) -> None:
        """Prepare backend state before forwarding one inference request.

        Most backends need no request-time transition. Stateful runtimes can
        override this hook, for example to activate an optional multimodal
        projector before the request reaches the worker.
        """
        return None

    def transform_request(self, model_id: str, path: str, body: dict) -> dict:
        """Return the body to POST to the worker, possibly transformed.

        The real dispatch path (``WorkerPool._prepare_forward``) POSTs the
        returned body to the worker. Backends whose worker speaks a different
        shape than the OpenAI/Ollama body override this to translate it. The
        default returns the body unchanged.

        This replaced the old ``forward_request``/``forward_stream`` methods,
        which the dispatch path never called (it POSTs directly) — so those
        were dead code that 16 test files kept validating. A backend that needs
        a genuinely different protocol (poll loop, multipart, sidecar) exposes
        a named method and the API layer calls it (see AceStep, whisper.cpp).
        """
        return body

    def request_timeout(self) -> float | None:
        """Per-request upstream timeout in seconds, or ``None`` for the default.

        ``None`` means ``WorkerPool`` uses ``INFERENCE_REQUEST_TIMEOUT_SECONDS``
        (floored at 30 s), which is what every backend uses today.
        """
        return None

    def request_headers(self, model_id: str) -> dict[str, str]:
        """Return private headers used by WorkerPool for worker requests.

        The default is intentionally empty. Backends whose localhost worker
        requires internal authentication can override this without exposing
        its credential through the public request body.
        """
        return {}

    # ------------------------------------------------------------------
    # Auxiliary ports (BUG-44)
    # ------------------------------------------------------------------

    # Set by the owner of the worker port range (WorkerPool) so that ports a
    # backend opens on its own — probe servers, sidecars — come from the same
    # managed range (settings.worker_port_range_*) and never collide with a
    # worker port. Without it, the old behaviour stays: an ephemeral port.
    _port_assign: Callable[[], Awaitable[int]] | None = None
    _port_release: Callable[[int], None] | None = None

    def attach_port_allocator(
        self,
        assign: Callable[[], Awaitable[int]],
        release: Callable[[int], None],
    ) -> None:
        """Route :meth:`_acquire_aux_port` through the managed port range."""
        self._port_assign = assign
        self._port_release = release

    async def _acquire_aux_port(self) -> int:
        """A free localhost port for a process the backend starts itself.

        Callers must hand it back with :meth:`_release_aux_port`.
        """
        if self._port_assign is not None:
            return await self._port_assign()
        return _ephemeral_port()

    def _release_aux_port(self, port: int) -> None:
        if self._port_release is not None and port:
            self._port_release(port)

    # ------------------------------------------------------------------
    # Multi-modal interface (optional — subclasses override as needed)
    # ------------------------------------------------------------------

    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        """Return the set of modalities this backend can handle.

        Subclasses MUST override this to declare their capabilities.
        """
        return set()

    async def upscale_video(
        self, model_id: str, video: bytes, **kwargs: Any
    ) -> tuple[bytes, dict[str, str]]:
        """Upscale/restore a video segment (``/ocabra/video/upscale``).

        Returns ``(encoded segment bytes, x-ocabra-* stats headers)``. Callers
        send one self-contained segment without audio and get the upscaled
        segment back; the backend keeps no state between calls so the
        scheduler stays free to evict the model in between.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support video upscaling")

    async def upscale_image(
        self, model_id: str, image: bytes, **kwargs: Any
    ) -> tuple[bytes, str, dict[str, str]]:
        """Upscale one still image (``/v1/images/upscale``).

        Returns ``(image bytes, content type, x-ocabra-* stats headers)``. Only
        backends that advertise ``image_upscaling`` implement it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support image upscaling")


# ----------------------------------------------------------------------------
# Optional backend capabilities declared as Protocols (REF-04)
# ----------------------------------------------------------------------------
#
# ``ModelManager`` and ``api/internal/models.py`` reach for these by
# duck-typing (``getattr(backend, "...", default)`` / ``hasattr``), so they
# must stay optional — adding them to ``BackendInterface`` with a default
# would make every ``getattr`` succeed and change which branch runs. These
# Protocols give the duck-typing a name and a signature without touching the
# runtime shape of the backends, so callers (and readers) can see the contract
# a backend must satisfy to be used through each of those paths. They are not
# enforced at runtime (they describe, they don't register).


@runtime_checkable
class SupportsLaunchPreview(Protocol):
    """Backends whose ``load`` can be dry-run for the model editor.

    Called from ``api/internal/models.py`` to show the exact command line /
    engine args a configuration would produce without starting anything.
    Implemented by vLLM and the llama.cpp family.
    """

    async def preview_launch(
        self, model_id: str, gpu_indices: list[int], extra_config: dict[str, Any]
    ) -> dict[str, Any]: ...


@runtime_checkable
class SupportsMemoryProfile(Protocol):
    """Backends that can run a real memory probe (``run_probe=True``).

    Called from ``api/internal/models.py`` to validate VRAM/KV with the engine
    itself. Implemented by vLLM.
    """

    async def estimate_memory_profile(
        self, model_id: str, *, gpu_index: int, extra_config: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


@runtime_checkable
class SupportsVisionState(Protocol):
    """Backends that load/drop a multimodal projector at runtime.

    ``vision_state`` reports whether the projector is resident right now, which
    only makes sense where vision is not fixed at launch (the llama.cpp fork).
    Read from ``api/internal/models.py``.
    """

    def vision_state(self, model_id: str) -> str | None: ...


@runtime_checkable
class SupportsLiveContext(Protocol):
    """Backends that recreate a worker's context in place, keeping its weights.

    ``SUPPORTS_LIVE_CONTEXT`` gates the live-context paths in ``ModelManager``
    and ``core/context_probe_service.py``; ``nominal_context_options`` reports
    the sizes a worker started below its maximum may grow back to. The fork
    backend (``llamacpp_mks``) sets the flag True.
    """

    SUPPORTS_LIVE_CONTEXT: bool

    def nominal_context_options(self, model_id: str) -> dict[str, int | None] | None: ...


@runtime_checkable
class ResolvesModelFile(Protocol):
    """Backends that can resolve a model id to its on-disk weight file.

    Consumed from ``ModelManager`` for disk-size estimates. Implemented by the
    llama.cpp family, BitNet and whisper.cpp.
    """

    def _resolve_model_file(self, model_id: str, extra_config: dict[str, Any]) -> Path: ...


@runtime_checkable
class DetectsRequestImages(Protocol):
    """Backends that can tell whether a request body carries an image.

    Consumed from ``ModelManager`` for vision admission. Implemented by the
    llama.cpp family.
    """

    @staticmethod
    def _contains_image(value: Any) -> bool: ...
