import asyncio
import json
from pathlib import Path

from ocabra.core.gguf_reader import read_gguf_header
from ocabra.schemas.registry import LocalModel

# --- GGUF header extraction (DUP-03) ---
# Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
# The single header parser lives in ``ocabra.core.gguf_reader``; the readers
# below are thin extractors over its cached result. Each is best-effort: a
# missing/unreadable file yields the same "unknown" it did before.


def parse_gguf_tokenizer_fingerprint(
    path: Path,
) -> tuple[int | None, int | None, int | None]:
    """Parse a GGUF file header and extract the tokenizer fingerprint.

    Returns ``(vocab_size, bos_id, eos_id)``. Any field may be ``None`` when
    the metadata is missing or unparseable. Errors are swallowed and result
    in an all-``None`` tuple.
    """
    header = read_gguf_header(path)
    if header is None:
        return (None, None, None)
    vocab_size = header.array_lengths.get("tokenizer.ggml.tokens")
    bos = header.metadata.get("tokenizer.ggml.bos_token_id")
    eos = header.metadata.get("tokenizer.ggml.eos_token_id")
    bos_id = int(bos) if isinstance(bos, (int, float)) else None
    eos_id = int(eos) if isinstance(eos, (int, float)) else None
    return (vocab_size, bos_id, eos_id)


def parse_gguf_file_type(path: Path) -> int | None:
    """``general.file_type`` of a GGUF, or ``None`` if unreadable.

    This is the quantization the file was produced with, and it is what tells
    apart builds that can load it: ``Q1_0`` (40) and ``Q2_0`` (41) only exist
    in the PrismML fork, for instance. The name cannot be trusted for this —
    an Ollama blob is called after its sha256.
    """
    header = read_gguf_header(path)
    if header is None:
        return None
    valor = header.metadata.get("general.file_type")
    return int(valor) if isinstance(valor, (int, float)) else None


def parse_gguf_context_length(path: Path) -> int | None:
    """Native context length declared by a GGUF, or ``None``.

    The key is namespaced by architecture (``qwen35moe.context_length``,
    ``flint.context_length``, ...), so ``general.architecture`` has to be read
    first. Used when registering a model to pick a context the model actually
    supports instead of llama.cpp's 4096 fallback.
    """
    header = read_gguf_header(path)
    if header is None:
        return None
    arch = header.metadata.get("general.architecture")
    arch = arch if isinstance(arch, str) else None
    contextos = {
        key: int(value)
        for key, value in header.metadata.items()
        if key.endswith(".context_length") and isinstance(value, (int, float))
    }
    if arch and f"{arch}.context_length" in contextos:
        return contextos[f"{arch}.context_length"]
    # Un solo candidato es inequivoco aunque no se haya leido la arquitectura.
    if len(contextos) == 1:
        return next(iter(contextos.values()))
    return None


def parse_gguf_chat_template(path: Path) -> str | None:
    """The Jinja chat template a GGUF ships (``tokenizer.chat_template``), or ``None``.

    Lets the model list show the tool and reasoning support a llama.cpp model
    really has while it is not loaded. Without it the list fell back to a name
    heuristic: flint:40b showed ``tools: false, reasoning: false`` although its
    template has ``<tool_call>`` and ``<think>`` (2026-11-11).
    """
    header = read_gguf_header(path)
    if header is None:
        return None
    template = header.metadata.get("tokenizer.chat_template")
    return template if isinstance(template, str) else None


def gguf_template_capabilities(path: Path) -> dict[str, bool] | None:
    """``{"tools", "reasoning"}`` support declared by a GGUF's chat template.

    Same markers ``LlamaCppBackend._template_caps`` reads from a running worker.
    """
    template = (parse_gguf_chat_template(path) or "").lower()
    if not template:
        return None
    return {
        "tools": any(m in template for m in ("tool_call", "tool_calls", "tools")),
        "reasoning": any(m in template for m in ("<think>", "</think>", "reasoning_content")),
    }


def gguf_reasoning_controls(path: Path) -> dict[str, bool] | None:
    """What a GGUF's chat template does with oCabra's reasoning controls.

    ``effort_levels``: the template reads ``reasoning_effort`` (levels change
    how much it thinks). ``thinking_switch``: it reads ``enable_thinking``
    (thinking can be turned on/off). A template with neither ignores the
    model's default reasoning effort entirely.
    """
    template = parse_gguf_chat_template(path) or ""
    if not template:
        return None
    return {
        "effort_levels": "reasoning_effort" in template,
        "thinking_switch": "enable_thinking" in template,
    }


# --- Which --spec-type a draft GGUF needs ---------------------------------
# A draft model is useless to llama-server unless it is also told *how* to
# drive it, and a draft with no type is not "no speculation": the server loads
# it anyway and aborts while probing its context. llama.cpp infers the type
# itself (``common_speculative_types_from_gguf``) but only recognises the
# exact tensor ``blk.<last>.nextn.eh_proj.weight``, or the ``dflash``
# architecture. A Gemma 4 assistant head has neither — its MTP block is
# ``nextn.pre_projection``/``nextn.post_projection`` — so the inference comes
# up empty and the worker dies on startup (``gemma4:12b``, 2026-11-08: every
# load since the draft layer appeared in its Ollama tag exited with SIGABRT
# and the model answered 503 for hours).
#
# Matching the ``nextn`` block by name instead of by one exact tensor covers
# both shapes.
_NEXTN_TENSOR_MARK = "nextn."
_DFLASH_ARCH = "dflash"
_DSPARK_TENSOR = "markov_w1.weight"


def parse_gguf_speculative_type(path: Path) -> str | None:
    """Return the ``--spec-type`` *path* needs as a draft model, or ``None``.

    ``None`` means "cannot tell from the file": callers must then not pass the
    draft to llama-server at all, since the alternative is a command line that
    kills the worker on startup.
    """

    header = read_gguf_header(path)
    if header is None:
        return None
    arch = header.metadata.get("general.architecture")
    arch = arch if isinstance(arch, str) else None
    tiene_nextn = any(_NEXTN_TENSOR_MARK in name for name in header.tensor_names)
    tiene_markov = any(name == _DSPARK_TENSOR for name in header.tensor_names)

    if arch == _DFLASH_ARCH:
        # The Markov head is what tells DSpark apart from DFlash.
        return "draft-dspark" if tiene_markov else "draft-dflash"
    if tiene_nextn:
        return "draft-mtp"
    return None


class LocalScanner:
    # Walking 200+ dirs and reading 14 × 64 MB of GGUF headers takes 15-25s on
    # this host. The Settings and Models pages both consume the scan and the
    # underlying tree only changes when the admin downloads / removes a model,
    # so we cache the result for a short TTL and additionally memoise the GGUF
    # fingerprint by (path, mtime, size) to avoid re-reading headers that
    # haven't changed.
    # Walking 200+ dirs and reading GGUF headers takes 15-25s on this host, so
    # the scan must never block a page render more than once. Within _SCAN_TTL_S
    # the cache is "fresh"; once stale we serve the stale result immediately and
    # refresh in the background (stale-while-revalidate), so only the very first
    # call ever pays the full scan latency.
    _SCAN_TTL_S = 120.0

    def __init__(self) -> None:
        self._scan_cache: tuple[float, tuple[str, str | None], list[LocalModel]] | None = None
        self._scan_lock = asyncio.Lock()
        self._refresh_task: asyncio.Task | None = None
        # (path, mtime_ns, size) -> (vocab_size, bos_id, eos_id)
        self._gguf_fingerprint_cache: dict[
            tuple[str, int, int], tuple[int | None, int | None, int | None]
        ] = {}

    def invalidate(self) -> None:
        """Drop the cached scan result. Call after a download/delete."""
        self._scan_cache = None

    def _schedule_refresh(
        self, models_dir: Path, ollama_shared_dir: Path | None, cache_key: tuple[str, str | None]
    ) -> None:
        """Kick off a background re-scan if one isn't already running."""
        if self._refresh_task is not None and not self._refresh_task.done():
            return

        async def _refresh() -> None:
            async with self._scan_lock:
                result = await asyncio.to_thread(self._scan_sync, models_dir, ollama_shared_dir)
                self._scan_cache = (asyncio.get_event_loop().time(), cache_key, result)

        self._refresh_task = asyncio.create_task(_refresh())

    async def scan(
        self, models_dir: Path, ollama_shared_dir: Path | None = None
    ) -> list[LocalModel]:
        cache_key = (str(models_dir), str(ollama_shared_dir) if ollama_shared_dir else None)

        # Fast path: serve from cache without acquiring the lock.
        cached = self._scan_cache
        if cached is not None and cached[1] == cache_key:
            stamp, _key, result = cached
            if (asyncio.get_event_loop().time() - stamp) < self._SCAN_TTL_S:
                return result
            # Stale: serve immediately, refresh in the background.
            self._schedule_refresh(models_dir, ollama_shared_dir, cache_key)
            return result

        # Slow path (no usable cache): lock so concurrent callers coalesce.
        async with self._scan_lock:
            cached = self._scan_cache
            if cached is not None and cached[1] == cache_key:
                stamp, _key, result = cached
                if (asyncio.get_event_loop().time() - stamp) < self._SCAN_TTL_S:
                    return result

            result = await asyncio.to_thread(self._scan_sync, models_dir, ollama_shared_dir)
            self._scan_cache = (asyncio.get_event_loop().time(), cache_key, result)
            return result

    def _scan_sync(
        self, models_dir: Path, ollama_shared_dir: Path | None = None
    ) -> list[LocalModel]:
        models: list[LocalModel] = []

        # Ollama-shared blobs run first so they appear next to the user's
        # downloads in the explore listing.
        if ollama_shared_dir is not None:
            models.extend(self._scan_ollama_shared(ollama_shared_dir))

        if not models_dir.exists():
            return models

        for path in models_dir.rglob("*"):
            if path.is_dir() and (path / "config.json").exists():
                size = self._dir_size(path)
                backend_type = self._detect_hf_backend(path)
                models.append(
                    LocalModel(
                        model_ref=path.name,
                        path=str(path),
                        source="huggingface",
                        backend_type=backend_type,
                        size_gb=size,
                    )
                )
                continue

            if path.is_dir() and (path / "Modelfile").exists():
                size = self._dir_size(path)
                models.append(
                    LocalModel(
                        model_ref=path.name,
                        path=str(path),
                        source="ollama",
                        backend_type="ollama",
                        size_gb=size,
                    )
                )
                continue

            if path.is_file() and path.suffix.lower() == ".gguf":
                # GGUF is the native llama.cpp format; default to llamacpp_mks,
                # the Makespace fork that production runs. Bitnet quantizations
                # (i2_s, "bitnet" in name) need their specialised backend.
                backend_type = "bitnet" if self._is_bitnet_gguf(path) else "llamacpp_mks"
                # Tokenizer fingerprint requires reading up to 64 MB from disk
                # per file. Memoise by (path, mtime_ns, size) so re-scans don't
                # re-hit the disk for files that haven't changed.
                try:
                    st = path.stat()
                    fp_key = (str(path), st.st_mtime_ns, st.st_size)
                except OSError:
                    fp_key = None
                cached_fp = self._gguf_fingerprint_cache.get(fp_key) if fp_key else None
                if cached_fp is not None:
                    vocab_size, bos_id, eos_id = cached_fp
                else:
                    vocab_size, bos_id, eos_id = parse_gguf_tokenizer_fingerprint(path)
                    if fp_key is not None:
                        self._gguf_fingerprint_cache[fp_key] = (vocab_size, bos_id, eos_id)
                models.append(
                    LocalModel(
                        model_ref=path.stem,
                        path=str(path),
                        source="gguf",
                        backend_type=backend_type,
                        size_gb=path.stat().st_size / (1024**3),
                        vocab_size=vocab_size,
                        bos_id=bos_id,
                        eos_id=eos_id,
                    )
                )
                continue

        unique: dict[str, LocalModel] = {model.path: model for model in models}
        return sorted(unique.values(), key=lambda m: m.model_ref)

    def _dir_size(self, root: Path) -> float | None:
        total = 0
        for child in root.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
        return total / (1024**3) if total > 0 else None

    def _detect_hf_backend(self, path: Path) -> str:
        """Detect the appropriate backend for a HuggingFace model directory.

        Checks config.json and directory name for known model families
        (Chatterbox, Whisper, TTS, Diffusers) and returns the matching
        backend type.  Defaults to ``vllm``.
        """
        import json

        name_lower = path.name.lower()

        # Chatterbox TTS models
        if "chatterbox" in name_lower:
            return "chatterbox"

        # Piper TTS voices (CPU-only VITS, rhasspy/piper-voices naming)
        if "piper" in name_lower:
            return "piper"

        # Whisper / faster-whisper models
        if "whisper" in name_lower:
            return "whisper"

        # VibeASR (microsoft/VibeVoice-ASR-BitNet) — ternary STT served by the
        # native asr_stream_server, distinct from faster-whisper.
        if "vibeasr" in name_lower or "vibevoice-asr" in name_lower:
            return "vibeasr"

        try:
            config = json.loads((path / "config.json").read_text())
        except Exception:
            config = {}

        model_type = str(config.get("model_type", "")).lower()

        if model_type == "whisper":
            return "whisper"

        if model_type in {"vibeasr", "vibevoice_asr", "vibe_voice_asr"}:
            return "vibeasr"

        # Mage-Flow ships a diffusers-style tree but a custom pipeline class,
        # so it needs its own backend (checked before the generic diffusers
        # rule below, which any model_index.json would otherwise match).
        model_index = path / "model_index.json"
        if model_index.exists():
            try:
                if (
                    str(json.loads(model_index.read_text()).get("_class_name", ""))
                    == "MageFlowPipeline"
                ):
                    return "mage"
            except Exception:
                pass

        # Diffusion models (e.g., Stable Diffusion, SDXL, Flux)
        if model_type in {"stable-diffusion", "sdxl"} or (path / "model_index.json").exists():
            return "diffusers"

        # TTS models (Kokoro, Bark, etc.)
        if "kokoro" in name_lower or "bark" in name_lower:
            return "tts"

        return "vllm"

    def _is_bitnet_gguf(self, path: Path) -> bool:
        name = path.name.lower()
        # dspark files are speculative-decoding drafters and mmproj files are
        # vision projectors — neither is a servable stand-alone LM.
        if "dspark" in name or "mmproj" in name:
            return False
        # bonsai / q1_0 → Bonsai (PrismML llama.cpp fork); i2_s → Microsoft BitNet.
        if any(marker in name for marker in ("bitnet", "i2_s", "bonsai", "q1_0")):
            return True

        # Best-effort header probe: avoid full file scan.
        try:
            with path.open("rb") as file:
                head = file.read(32768).lower()
        except OSError:
            return False
        return b"bitnet" in head or b"i2_s" in head or b"bonsai" in head

    def _scan_ollama_shared(self, root: Path) -> list[LocalModel]:
        """Surface Ollama-pulled models as additional ``llama_cpp`` candidates.

        Ollama stores each pulled model as an OCI manifest in
        ``manifests/<registry>/<owner>/<model>/<tag>`` plus content-addressed
        blobs under ``blobs/sha256-<digest>``. The blob whose layer mediaType
        is ``application/vnd.ollama.image.model`` is a stock GGUF and can be
        passed directly to llama-server. This lets the user reuse models
        already pulled via Ollama without a second download or registration.

        Best-effort: any IO/JSON error is silently dropped per manifest.
        """
        if not root.exists():
            return []

        manifests_root = root / "manifests"
        blobs_root = root / "blobs"
        if not manifests_root.exists() or not blobs_root.exists():
            return []

        out: list[LocalModel] = []
        for manifest in manifests_root.rglob("*"):
            if not manifest.is_file():
                continue
            try:
                payload = json.loads(manifest.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            layers = payload.get("layers")
            if not isinstance(layers, list):
                continue
            model_layer = next(
                (
                    layer
                    for layer in layers
                    if isinstance(layer, dict)
                    and layer.get("mediaType") == "application/vnd.ollama.image.model"
                ),
                None,
            )
            if model_layer is None:
                continue
            digest = str(model_layer.get("digest", ""))
            if not digest.startswith("sha256:"):
                continue
            blob_filename = "sha256-" + digest.split(":", 1)[1]
            blob_path = blobs_root / blob_filename
            if not blob_path.is_file():
                continue
            # Manifest path: .../manifests/<registry>/<owner>/<model>/<tag>
            try:
                tag = manifest.name
                model_name = manifest.parent.name
            except Exception:
                continue
            model_ref = f"{model_name}:{tag}"
            vocab_size, bos_id, eos_id = parse_gguf_tokenizer_fingerprint(blob_path)
            out.append(
                LocalModel(
                    model_ref=model_ref,
                    path=str(blob_path),
                    source="ollama-shared",
                    backend_type="llamacpp_mks",
                    size_gb=blob_path.stat().st_size / (1024**3),
                    vocab_size=vocab_size,
                    bos_id=bos_id,
                    eos_id=eos_id,
                )
            )
        return out
