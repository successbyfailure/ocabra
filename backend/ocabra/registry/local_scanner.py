import asyncio
import json
import struct
from pathlib import Path

from ocabra.schemas.registry import LocalModel

# --- GGUF binary header parsing (Sprint 17.4) ---
# Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
# We only extract the tokenizer fingerprint (vocab_size, bos_id, eos_id) so we
# can decide whether two GGUFs are speculative-compatible. A best-effort parse
# that returns ``(None, None, None)`` on any error is acceptable.

_GGUF_MAGIC = b"GGUF"

# GGUF metadata value types (subset we care about).
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

_GGUF_SCALAR_FORMATS: dict[int, str] = {
    _GGUF_TYPE_UINT8: "<B",
    _GGUF_TYPE_INT8: "<b",
    _GGUF_TYPE_UINT16: "<H",
    _GGUF_TYPE_INT16: "<h",
    _GGUF_TYPE_UINT32: "<I",
    _GGUF_TYPE_INT32: "<i",
    _GGUF_TYPE_FLOAT32: "<f",
    _GGUF_TYPE_BOOL: "<B",
    _GGUF_TYPE_UINT64: "<Q",
    _GGUF_TYPE_INT64: "<q",
    _GGUF_TYPE_FLOAT64: "<d",
}

_TOKENIZER_KEYS = {
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
}

# Cap the bytes we read from each GGUF — kv_data lives at the top of the file
# but huge tokenizer arrays (e.g. 256k vocab) can push it past 32 MiB. 64 MiB
# is a safe upper bound that avoids slurping multi-GB model files.
_GGUF_HEADER_MAX_BYTES = 64 * 1024 * 1024


class _GGUFParseError(Exception):
    """Raised when a GGUF header cannot be parsed."""


def _read_scalar(buf: memoryview, offset: int, value_type: int) -> tuple[object, int]:
    fmt = _GGUF_SCALAR_FORMATS[value_type]
    size = struct.calcsize(fmt)
    (value,) = struct.unpack_from(fmt, buf, offset)
    if value_type == _GGUF_TYPE_BOOL:
        value = bool(value)
    return value, offset + size


def _read_string(buf: memoryview, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", buf, offset)
    offset += 8
    raw = bytes(buf[offset : offset + length])
    return raw.decode("utf-8", errors="replace"), offset + length


def _skip_value(buf: memoryview, offset: int, value_type: int) -> int:
    if value_type in _GGUF_SCALAR_FORMATS:
        _, offset = _read_scalar(buf, offset, value_type)
        return offset
    if value_type == _GGUF_TYPE_STRING:
        _, offset = _read_string(buf, offset)
        return offset
    if value_type == _GGUF_TYPE_ARRAY:
        (item_type,) = struct.unpack_from("<I", buf, offset)
        offset += 4
        (count,) = struct.unpack_from("<Q", buf, offset)
        offset += 8
        for _ in range(count):
            offset = _skip_value(buf, offset, int(item_type))
        return offset
    raise _GGUFParseError(f"unknown gguf type {value_type}")


def _read_array_count(buf: memoryview, offset: int) -> tuple[int, int, int]:
    (item_type,) = struct.unpack_from("<I", buf, offset)
    offset += 4
    (count,) = struct.unpack_from("<Q", buf, offset)
    offset += 8
    return int(item_type), int(count), offset


def parse_gguf_tokenizer_fingerprint(
    path: Path,
) -> tuple[int | None, int | None, int | None]:
    """Parse a GGUF file header and extract the tokenizer fingerprint.

    Returns ``(vocab_size, bos_id, eos_id)``. Any field may be ``None`` when
    the metadata is missing or unparseable. Errors are swallowed and result
    in an all-``None`` tuple.
    """
    try:
        with path.open("rb") as fh:
            data = fh.read(_GGUF_HEADER_MAX_BYTES)
    except OSError:
        return (None, None, None)

    if len(data) < 24 or data[:4] != _GGUF_MAGIC:
        return (None, None, None)

    buf = memoryview(data)
    try:
        (version,) = struct.unpack_from("<I", buf, 4)
        if version < 2:
            # v1 used 32-bit lengths; we only support v2+.
            return (None, None, None)
        (_tensor_count,) = struct.unpack_from("<Q", buf, 8)
        (kv_count,) = struct.unpack_from("<Q", buf, 16)
        offset = 24

        vocab_size: int | None = None
        bos_id: int | None = None
        eos_id: int | None = None

        for _ in range(int(kv_count)):
            key, offset = _read_string(buf, offset)
            (value_type,) = struct.unpack_from("<I", buf, offset)
            offset += 4
            value_type = int(value_type)

            if key not in _TOKENIZER_KEYS:
                offset = _skip_value(buf, offset, value_type)
                continue

            if key == "tokenizer.ggml.tokens":
                if value_type != _GGUF_TYPE_ARRAY:
                    offset = _skip_value(buf, offset, value_type)
                    continue
                item_type, count, offset = _read_array_count(buf, offset)
                vocab_size = count
                # Skip the actual token strings/IDs — we only need the count.
                for _ in range(count):
                    offset = _skip_value(buf, offset, item_type)
            elif key == "tokenizer.ggml.bos_token_id":
                if value_type in _GGUF_SCALAR_FORMATS:
                    value, offset = _read_scalar(buf, offset, value_type)
                    bos_id = int(value) if value is not None else None
                else:
                    offset = _skip_value(buf, offset, value_type)
            elif key == "tokenizer.ggml.eos_token_id":
                if value_type in _GGUF_SCALAR_FORMATS:
                    value, offset = _read_scalar(buf, offset, value_type)
                    eos_id = int(value) if value is not None else None
                else:
                    offset = _skip_value(buf, offset, value_type)

            if vocab_size is not None and bos_id is not None and eos_id is not None:
                break

        return (vocab_size, bos_id, eos_id)
    except (struct.error, _GGUFParseError, IndexError, ValueError):
        return (None, None, None)


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
        self._gguf_fingerprint_cache: dict[tuple[str, int, int], tuple[int | None, int | None, int | None]] = {}

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
                # GGUF is the native llama.cpp format; default to llama_cpp.
                # Bitnet quantizations (i2_s, "bitnet" in name) need their
                # specialised backend.
                backend_type = "bitnet" if self._is_bitnet_gguf(path) else "llama_cpp"
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

        # Whisper / faster-whisper models
        if "whisper" in name_lower:
            return "whisper"

        try:
            config = json.loads((path / "config.json").read_text())
        except Exception:
            config = {}

        model_type = str(config.get("model_type", "")).lower()

        if model_type == "whisper":
            return "whisper"

        # Diffusion models (e.g., Stable Diffusion, SDXL, Flux)
        if model_type in {"stable-diffusion", "sdxl"} or (path / "model_index.json").exists():
            return "diffusers"

        # TTS models (Kokoro, Bark, etc.)
        if "kokoro" in name_lower or "bark" in name_lower:
            return "tts"

        return "vllm"

    def _is_bitnet_gguf(self, path: Path) -> bool:
        name = path.name.lower()
        if "bitnet" in name or "i2_s" in name:
            return True

        # Best-effort header probe: avoid full file scan.
        try:
            with path.open("rb") as file:
                head = file.read(32768).lower()
        except OSError:
            return False
        return b"bitnet" in head or b"i2_s" in head

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
                    backend_type="llama_cpp",
                    size_gb=blob_path.stat().st_size / (1024**3),
                    vocab_size=vocab_size,
                    bos_id=bos_id,
                    eos_id=eos_id,
                )
            )
        return out
