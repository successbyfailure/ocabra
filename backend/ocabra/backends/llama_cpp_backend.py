from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import structlog

from ocabra.backends.base import (
    BackendCapabilities,
    BackendInstallSpec,
    BackendInterface,
    ModalityType,
    WorkerInfo,
)
from ocabra.config import settings
from ocabra.core.backend_installer import read_backend_metadata

logger = structlog.get_logger(__name__)

_DEFAULT_TOTAL_LAYERS = 32
_DEFAULT_STARTUP_TIMEOUT_S = 30
_SHUTDOWN_TIMEOUT_S = 20
_WORKER_PATH = Path(__file__).resolve().parents[1] / "workers" / "llama_cpp_worker.py"
_VALID_SPLIT_MODES = {"layer", "row", "none"}
_VALID_SPLIT_STRATEGIES = {"evenly", "favor_main"}


def _compose_visible_devices(
    preferred_gpu: list[int],
    disabled_gpus: list[int] | None,
    gpu_manager: Any | None = None,
) -> list[int]:
    """Compose the final list of GPU indices visible to the worker.

    Args:
        preferred_gpu: GPU indices assigned by the scheduler / caller. Order
            is preserved because llama.cpp interprets index 0 as the first
            visible device, which interacts with ``--main-gpu``.
        disabled_gpus: indices the user explicitly asked to keep free. They
            are removed from the result regardless of source.
        gpu_manager: optional GPU manager used to validate that disabled
            indices exist. The manager is read-only here; the function never
            mutates state and tolerates ``None`` so unit tests can omit it.

    Returns:
        The filtered list of GPU indices, preserving the input order.
    """

    blocked = set(disabled_gpus or ())
    if gpu_manager is not None and blocked:
        # Best-effort sanity check: silently drop indices the manager does
        # not know about so we never produce an empty CUDA_VISIBLE_DEVICES
        # by accident.
        known = getattr(gpu_manager, "_states", None)
        if isinstance(known, dict):
            blocked = {idx for idx in blocked if idx in known}
    return [idx for idx in preferred_gpu if idx not in blocked]


class LlamaCppBackend(BackendInterface):
    @classmethod
    def supported_modalities(cls) -> set[ModalityType]:
        return {ModalityType.TEXT_GENERATION, ModalityType.EMBEDDINGS}

    @property
    def install_spec(self) -> BackendInstallSpec:
        """Modular install spec for llama.cpp (Bloque 15 Fase 2).

        Native build: clones ggml-org/llama.cpp and runs cmake (CUDA-enabled
        when the host has the toolkit, CPU otherwise). The produced
        ``llama-server`` binary lives under ``<backend_dir>/bin/`` and is
        resolved at load time via ``_resolve_server_bin()``.
        """

        return BackendInstallSpec(
            oci_image="ghcr.io/ocabra/backend-llama-cpp",
            oci_tags={"cuda12": "latest-cuda12", "cpu": "latest-cpu"},
            apt_packages=[
                "build-essential",
                "cmake",
                "git",
                "ninja-build",
                "ca-certificates",
            ],
            git_repo="https://github.com/ggml-org/llama.cpp",
            git_ref="master",
            post_install_script="backend/scripts/install_llama_cpp.sh",
            extra_bins={"server": "bin/llama-server"},
            include_core_runtime=False,
            estimated_size_mb=300,
            display_name="llama.cpp",
            description=(
                "Native llama.cpp server with CUDA acceleration. Supports GGUF "
                "models for text generation and embeddings."
            ),
            tags=["LLM", "GGUF", "CUDA"],
        )

    def __init__(self, gpu_manager: Any | None = None) -> None:
        self._processes: dict[str, tuple[asyncio.subprocess.Process, int]] = {}
        self._model_configs: dict[str, dict[str, Any]] = {}
        self._gpu_manager = gpu_manager

    def set_gpu_manager(self, gpu_manager: Any) -> None:
        """Inject the :class:`~ocabra.core.gpu_manager.GPUManager`.

        The backend only reads VRAM totals to compute ``tensor_split`` ratios
        when ``split_strategy='evenly'``. It never mutates the manager.
        """

        self._gpu_manager = gpu_manager

    def _resolve_server_bin(self) -> str:
        """Pick the ``llama-server`` binary path.

        Priority:
        1. ``<backends_dir>/llama_cpp/metadata.json`` -> ``extra_bins.server``
           (modular install via :class:`BackendInstaller`).
        2. ``settings.llama_cpp_server_bin`` (legacy fat-image path).
        """
        meta = read_backend_metadata(settings.backends_dir, "llama_cpp")
        if meta is not None:
            extra = meta.get("extra_bins") if isinstance(meta, dict) else None
            if isinstance(extra, dict):
                bin_path = extra.get("server")
                if bin_path and Path(bin_path).is_file():
                    return str(bin_path)
        return settings.llama_cpp_server_bin

    async def load(self, model_id: str, gpu_indices: list[int], **kwargs) -> WorkerInfo:
        port = int(kwargs.get("port") or 0)
        if port == 0:
            raise ValueError("load() requires 'port' kwarg — assign via WorkerPool.assign_port()")
        if not _WORKER_PATH.exists():
            raise FileNotFoundError(f"llama_cpp_worker.py not found at '{_WORKER_PATH}'")

        extra_config = kwargs.get("extra_config") or {}
        model_file = self._resolve_model_file(model_id, extra_config)
        options = self._build_options(extra_config)
        options["model_file"] = str(model_file)

        visible_gpus = _compose_visible_devices(
            gpu_indices,
            options.get("disabled_gpus"),
            self._gpu_manager,
        )
        # Auto-compute tensor_split when the user picked "evenly" and did not
        # supply explicit ratios. We need at least two visible GPUs for it to
        # make sense.
        if (
            options.get("tensor_split") is None
            and options.get("split_strategy") == "evenly"
            and len(visible_gpus) > 1
        ):
            options["tensor_split"] = await self._compute_evenly_tensor_split(visible_gpus)

        self._model_configs[model_id] = options

        cmd = [
            sys.executable,
            str(_WORKER_PATH),
            "--server-bin",
            self._get_binary_path(options.get("runtime")),
            "--model-id",
            model_id,
            "--model-path",
            str(model_file),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(options["ctx_size"]),
            "--batch-size",
            str(options["batch_size"]),
            "--ubatch-size",
            str(options["ubatch_size"]),
            "--gpu-layers",
            str(options["gpu_layers"]),
            "--alias",
            model_id,
        ]
        if options["threads"] is not None:
            cmd.extend(["--threads", str(options["threads"])])
        if options["flash_attn"]:
            cmd.append("--flash-attn")
        if options["mlock"]:
            cmd.append("--mlock")
        if options["embedding"]:
            cmd.append("--embedding")
        # Sprint 17.1 — Tier 1 flags.
        # ``mmap`` is opt-out: False explicitly disables it.
        if options["mmap"] is False:
            cmd.append("--no-mmap")
        if options["seed"] is not None:
            cmd.extend(["--seed", str(options["seed"])])
        if options["no_kv_offload"]:
            cmd.append("--no-kv-offload")
        if options["rope_freq_base"] is not None:
            cmd.extend(["--rope-freq-base", str(options["rope_freq_base"])])
        if options["rope_freq_scale"] is not None:
            cmd.extend(["--rope-freq-scale", str(options["rope_freq_scale"])])
        # Sprint 17.2 — KV cache quantization
        if options.get("cache_type_k"):
            cmd.extend(["--cache-type-k", str(options["cache_type_k"])])
        if options.get("cache_type_v"):
            cmd.extend(["--cache-type-v", str(options["cache_type_v"])])

        # Multi-GPU + MoE flags (Sprint 17.3). All optional; only forwarded
        # when the user actually configured them.
        if options.get("main_gpu") is not None:
            cmd.extend(["--main-gpu", str(int(options["main_gpu"]))])
        if options.get("tensor_split"):
            cmd.extend(
                [
                    "--tensor-split",
                    ",".join(self._format_ratio(r) for r in options["tensor_split"]),
                ]
            )
        if options.get("split_mode") is not None:
            cmd.extend(["--split-mode", str(options["split_mode"])])
        if options.get("n_cpu_moe") is not None:
            cmd.extend(["--n-cpu-moe", str(int(options["n_cpu_moe"]))])
        if options.get("override_tensor"):
            cmd.extend(["--override-tensor", str(options["override_tensor"])])

        # --- Sprint 17.4 — speculative decoding + concurrency ---
        speculative = options.get("speculative")
        if isinstance(speculative, dict):
            draft_id = speculative.get("draft_model_id")
            if draft_id:
                draft_path = self._resolve_draft_model_path(str(draft_id))
                cmd.extend(["--model-draft", str(draft_path)])
                draft_n = speculative.get("draft_n")
                if draft_n is not None:
                    cmd.extend(["--draft-max", str(int(draft_n))])
                draft_min = speculative.get("draft_min")
                if draft_min is not None:
                    cmd.extend(["--draft-min", str(int(draft_min))])
                draft_p_min = speculative.get("draft_p_min")
                if draft_p_min is not None:
                    cmd.extend(["--draft-p-min", str(float(draft_p_min))])

        if options.get("parallel_slots") is not None:
            cmd.extend(["--parallel", str(int(options["parallel_slots"]))])
        if options.get("cont_batching"):
            cmd.append("--cont-batching")

        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": settings.cuda_device_order,
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in visible_gpus)
            if options["gpu_layers"] > 0
            else "",
        }

        logger.info(
            "llama_cpp_starting",
            model_id=model_id,
            port=port,
            gpu_layers=options["gpu_layers"],
        )
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._processes[model_id] = (process, port)

        try:
            await self._wait_for_startup(
                model_id,
                port,
                timeout_s=int(options.get("startup_timeout_s") or _DEFAULT_STARTUP_TIMEOUT_S),
            )
        except Exception:
            await self._kill_process(model_id)
            raise

        vram_used_mb = await self.get_vram_estimate_mb(model_id)
        return WorkerInfo(
            backend_type="llama_cpp",
            model_id=model_id,
            gpu_indices=visible_gpus or gpu_indices,
            port=port,
            pid=process.pid or 0,
            vram_used_mb=vram_used_mb,
        )

    async def unload(self, model_id: str) -> None:
        await self._kill_process(model_id, graceful=True)
        self._model_configs.pop(model_id, None)

    async def health_check(self, model_id: str) -> bool:
        entry = self._processes.get(model_id)
        if not entry:
            return False
        process, port = entry
        if process.returncode is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"http://127.0.0.1:{port}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def get_capabilities(self, model_id: str) -> BackendCapabilities:
        options = self._model_configs.get(model_id, {})
        normalized = model_id.lower()
        embeddings = bool(options.get("embedding")) or any(
            token in normalized for token in ("embed", "embedding", "bge", "e5")
        )
        vision = any(token in normalized for token in ("llava", "vision", "minicpmv", "vl"))
        reasoning = any(token in normalized for token in ("deepseek-r1", "qwen3", "reason"))
        tools = not embeddings and any(
            token in normalized for token in ("llama", "mistral", "mixtral", "qwen", "tool")
        )
        return BackendCapabilities(
            chat=not embeddings,
            completion=True,
            tools=tools,
            vision=vision,
            embeddings=embeddings,
            reasoning=reasoning,
            streaming=True,
            context_length=int(options.get("ctx_size", settings.llama_cpp_ctx_size)),
        )

    async def get_vram_estimate_mb(self, model_id: str) -> int:
        options = self._model_configs.get(model_id, {})
        gpu_layers = int(options.get("gpu_layers", settings.llama_cpp_gpu_layers))
        if gpu_layers <= 0:
            return 0

        total_layers = max(1, int(options.get("total_layers", _DEFAULT_TOTAL_LAYERS)))
        model_file = options.get("model_file")
        if model_file:
            size_bytes = await asyncio.to_thread(lambda: Path(model_file).stat().st_size)
            total_mb = max(1, int(size_bytes / (1024 * 1024)))
        else:
            total_mb = max(1, int(options.get("model_vram_mb", 4096)))
        return max(1, int(total_mb * min(gpu_layers, total_layers) / total_layers))

    async def forward_request(self, model_id: str, path: str, body: dict) -> Any:
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No llama.cpp worker for model '{model_id}'")
        _, port = entry
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"http://127.0.0.1:{port}{path}", json=body)
            response.raise_for_status()
            return response.json()

    async def forward_stream(self, model_id: str, path: str, body: dict) -> AsyncIterator[bytes]:
        entry = self._processes.get(model_id)
        if not entry:
            raise KeyError(f"No llama.cpp worker for model '{model_id}'")
        _, port = entry
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", f"http://127.0.0.1:{port}{path}", json=body
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    def _build_options(self, extra_config: dict[str, Any]) -> dict[str, Any]:
        options = {
            "gpu_layers": int(
                self._get_option(extra_config, "gpu_layers", settings.llama_cpp_gpu_layers)
            ),
            "ctx_size": int(
                self._get_option(extra_config, "ctx_size", settings.llama_cpp_ctx_size)
            ),
            "threads": self._to_int_or_none(
                self._get_option(extra_config, "threads", settings.llama_cpp_threads)
            ),
            "batch_size": int(
                self._get_option(extra_config, "batch_size", settings.llama_cpp_batch_size)
            ),
            "ubatch_size": int(
                self._get_option(extra_config, "ubatch_size", settings.llama_cpp_ubatch_size)
            ),
            "flash_attn": bool(
                self._get_option(extra_config, "flash_attn", settings.llama_cpp_flash_attn)
            ),
            "mlock": bool(self._get_option(extra_config, "mlock", settings.llama_cpp_mlock)),
            "embedding": bool(
                self._get_option(extra_config, "embedding", settings.llama_cpp_embeddings)
            ),
            "startup_timeout_s": int(
                self._get_option(
                    extra_config,
                    "startup_timeout_s",
                    settings.llama_cpp_startup_timeout_s,
                )
            ),
            "total_layers": int(
                self._get_option(extra_config, "total_layers", _DEFAULT_TOTAL_LAYERS)
            ),
            # --- Sprint 17.1 (Tier 1) flags ---
            # ``mmap`` is tri-state: None => use llama-server default (mmap on),
            # False => pass --no-mmap, True => explicit no-op (still on).
            "mmap": self._to_bool_or_none(self._get_option(extra_config, "mmap", None)),
            "seed": self._to_int_or_none(self._get_option(extra_config, "seed", None)),
            "no_kv_offload": bool(self._get_option(extra_config, "no_kv_offload", False)),
            "rope_freq_base": self._to_float_or_none(
                self._get_option(extra_config, "rope_freq_base", None)
            ),
            "rope_freq_scale": self._to_float_or_none(
                self._get_option(extra_config, "rope_freq_scale", None)
            ),
            # Sprint 17.2 — KV cache quantization (None = use llama-server default)
            "cache_type_k": self._get_option(extra_config, "cache_type_k", None),
            "cache_type_v": self._get_option(extra_config, "cache_type_v", None),
            # Multi-GPU + MoE (Sprint 17.3). All optional.
            "main_gpu": self._to_int_or_none(self._get_option(extra_config, "main_gpu", None)),
            "tensor_split": self._normalize_tensor_split(
                self._get_option(extra_config, "tensor_split", None)
            ),
            "split_mode": self._normalize_split_mode(
                self._get_option(extra_config, "split_mode", None)
            ),
            "disabled_gpus": self._normalize_int_list(
                self._get_option(extra_config, "disabled_gpus", None)
            ),
            "split_strategy": self._normalize_split_strategy(
                self._get_option(extra_config, "split_strategy", None)
            ),
            "n_cpu_moe": self._to_int_or_none(self._get_option(extra_config, "n_cpu_moe", None)),
            "override_tensor": self._normalize_str(
                self._get_option(extra_config, "override_tensor", None)
            ),
            # --- Sprint 17.4 ---
            "speculative": self._get_option(extra_config, "speculative", None),
            "runtime": self._get_option(extra_config, "runtime", None),
            "parallel_slots": self._to_int_or_none(
                self._get_option(extra_config, "parallel_slots", None)
            ),
            "cont_batching": self._to_bool_or_none(
                self._get_option(extra_config, "cont_batching", None)
            ),
            "keep_alive_seconds": self._to_int_or_none(
                self._get_option(extra_config, "keep_alive_seconds", None)
            ),
        }
        # Sprint 17.2 — quantized V cache requires flash attention.
        cache_v = options["cache_type_v"]
        if cache_v is not None and cache_v != "f16" and not options["flash_attn"]:
            raise ValueError(
                "cache_type_v != 'f16' requires flash_attn=True (llama.cpp only "
                "supports quantized V cache when flash attention is enabled)."
            )
        return options

    def _get_option(self, extra_config: dict[str, Any], key: str, default: Any) -> Any:
        # Accept both snake_case (canonical schema) and camelCase (frontend
        # write path) so persisted state from either side resolves correctly.
        camel_key = "".join(
            part.capitalize() if index else part for index, part in enumerate(key.split("_"))
        )
        nested = extra_config.get("llama_cpp")
        if isinstance(nested, dict):
            if key in nested:
                return nested[key]
            if camel_key in nested:
                return nested[camel_key]
        if key in extra_config:
            return extra_config[key]
        if camel_key in extra_config:
            return extra_config[camel_key]
        return default

    def _resolve_model_file(self, model_id: str, extra_config: dict[str, Any]) -> Path:
        explicit = self._get_option(extra_config, "model_path", None)
        if explicit:
            path = Path(str(explicit))
            if path.is_file():
                return path

        direct = Path(model_id)
        if direct.is_file():
            return direct

        root = Path(settings.models_dir)
        candidates = [
            root / model_id,
            root / f"{model_id}.gguf",
            root / "huggingface" / model_id.replace("/", "--"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        stem = Path(model_id).stem
        matches = [path for path in root.rglob("*.gguf") if path.stem == stem]
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime)

        raise FileNotFoundError(f"llama.cpp GGUF model not found for '{model_id}' under {root}")

    async def _wait_for_startup(self, model_id: str, port: int, timeout_s: int) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        url = f"http://127.0.0.1:{port}/health"
        while asyncio.get_running_loop().time() < deadline:
            entry = self._processes.get(model_id)
            if entry and entry[0].returncode is not None:
                rc = entry[0].returncode
                stderr_tail = await self._read_stderr_tail(entry[0])
                suffix = f" stderr={stderr_tail}" if stderr_tail else ""
                if rc in {-signal.SIGKILL, 137}:
                    raise MemoryError(
                        f"llama.cpp worker for '{model_id}' was OOM-killed (rc={rc}).{suffix}"
                    )
                raise RuntimeError(
                    f"llama.cpp worker for '{model_id}' exited with rc={rc}.{suffix}"
                )
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(url)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(1)
        raise TimeoutError(
            f"llama.cpp worker for '{model_id}' did not become healthy within {timeout_s}s"
        )

    async def _kill_process(self, model_id: str, graceful: bool = False) -> None:
        entry = self._processes.pop(model_id, None)
        if not entry:
            return
        process, _ = entry
        if process.returncode is not None:
            return

        if graceful:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=float(_SHUTDOWN_TIMEOUT_S))
                return
            except (TimeoutError, ProcessLookupError):
                logger.warning("llama_cpp_sigterm_timeout", model_id=model_id)

        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass

    async def _read_stderr_tail(
        self, process: asyncio.subprocess.Process, limit: int = 4000
    ) -> str:
        if process.stderr is None:
            return ""
        try:
            data = await asyncio.wait_for(process.stderr.read(), timeout=0.3)
        except Exception:
            return ""
        if not data:
            return ""
        text = data.decode("utf-8", errors="replace").strip()
        if len(text) <= limit:
            return text
        half = limit // 2
        return f"{text[:half]}\n...[truncated]...\n{text[-half:]}"

    def _to_int_or_none(self, value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    def _to_float_or_none(self, value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    def _to_bool_or_none(self, value: Any) -> bool | None:
        if value is None:
            return None
        return bool(value)

    def _normalize_tensor_split(self, value: Any) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
        elif isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            return None
        ratios: list[float] = []
        for raw in parts:
            try:
                ratios.append(float(raw))
            except (TypeError, ValueError):
                return None
        if not ratios or all(r == 0 for r in ratios) or any(r < 0 for r in ratios):
            return None
        return ratios

    def _normalize_int_list(self, value: Any) -> list[int] | None:
        if value is None:
            return None
        if isinstance(value, str):
            parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
        elif isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            return None
        out: list[int] = []
        seen: set[int] = set()
        for raw in parts:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx in seen:
                continue
            out.append(idx)
            seen.add(idx)
        return out or None

    def _normalize_split_mode(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        return text if text in _VALID_SPLIT_MODES else None

    def _normalize_split_strategy(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        return text if text in _VALID_SPLIT_STRATEGIES else None

    def _normalize_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _format_ratio(value: float) -> str:
        """Format a tensor-split ratio for the CLI.

        Uses an integer literal when the value is a whole number to keep the
        flag tidy (``--tensor-split 3,1`` rather than ``3.0,1.0``).
        """

        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):g}"

    async def _compute_evenly_tensor_split(self, gpu_indices: list[int]) -> list[float] | None:
        """Compute split ratios proportional to each GPU's total VRAM.

        Reads ``total_vram_mb`` from the injected GPU manager. Falls back to
        ``None`` (i.e. let llama-server decide) when the manager is missing
        or any GPU reports a non-positive total.
        """

        manager = self._gpu_manager
        if manager is None:
            return None
        ratios: list[float] = []
        for idx in gpu_indices:
            getter = getattr(manager, "get_state", None)
            state = None
            if callable(getter):
                try:
                    state = await getter(idx)
                except Exception:
                    state = None
            if state is None:
                state = (
                    manager.get_state_nowait(idx) if hasattr(manager, "get_state_nowait") else None
                )
            total = getattr(state, "total_vram_mb", 0) if state is not None else 0
            if total <= 0:
                return None
            ratios.append(float(total))
        # Normalize to the smallest GPU = 1 so the resulting CSV is compact
        # (``3,1`` instead of ``24576,8192``).
        smallest = min(ratios)
        if smallest <= 0:
            return None
        return [round(r / smallest, 4) for r in ratios]

    def _get_binary_path(self, runtime: str | None) -> str:
        """Resolve the ``llama-server`` binary path for a given runtime variant.

        Sprint 17.4 introduces optional ``runtime`` selection
        (``cuda``/``rocm``/``vulkan``/``cpu``). When ``runtime`` is ``None`` or
        ``"cuda"`` the existing default lookup is used. For alternate runtimes
        the installer is expected to have placed a binary under
        ``<backends_dir>/llama_cpp_<runtime>/bin/llama-server``; if missing we
        raise a clear error pointing to the ``/backends`` page.
        """
        if runtime is None or runtime == "cuda":
            return self._resolve_server_bin()

        backend_name = f"llama_cpp_{runtime}"
        meta = read_backend_metadata(settings.backends_dir, backend_name)
        if meta is not None and isinstance(meta, dict):
            extra = meta.get("extra_bins")
            if isinstance(extra, dict):
                bin_path = extra.get("server")
                if bin_path and Path(bin_path).is_file():
                    return str(bin_path)

        # Fallback: convention-based location.
        candidate = Path(settings.backends_dir) / backend_name / "bin" / "llama-server"
        if candidate.is_file():
            return str(candidate)

        raise FileNotFoundError(
            f"llama.cpp runtime '{runtime}' is not installed. "
            f"Install it from the /backends page (variant 'llama_cpp:{runtime}')."
        )

    def _resolve_draft_model_path(self, draft_model_id: str) -> Path:
        """Resolve a draft model_id into a local GGUF path.

        Strategy:
            1. If the value is an existing file path, use it.
            2. Otherwise treat it as a canonical model_id and re-use the
               regular ``_resolve_model_file`` lookup over ``settings.models_dir``.
        """
        direct = Path(draft_model_id)
        if direct.is_file():
            return direct
        return self._resolve_model_file(draft_model_id, {})
