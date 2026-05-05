import asyncio
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
    async def scan(self, models_dir: Path) -> list[LocalModel]:
        return await asyncio.to_thread(self._scan_sync, models_dir)

    def _scan_sync(self, models_dir: Path) -> list[LocalModel]:
        models: list[LocalModel] = []

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
                backend_type = "bitnet" if self._is_bitnet_gguf(path) else "vllm"
                vocab_size, bos_id, eos_id = parse_gguf_tokenizer_fingerprint(path)
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
