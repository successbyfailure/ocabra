from __future__ import annotations

import asyncio
import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ocabra.config import settings
from ocabra.schemas.registry import HFVLLMRuntimeProbe, HFVLLMSupport

_PROBE_STARTUP_TIMEOUT_S = 25
_PROBE_SHUTDOWN_TIMEOUT_S = 5


@dataclass
class _ProbeCommand:
    cmd: list[str]
    env: dict[str, str]
    target: str
    port: int


class VLLMRuntimeProbeService:
    def __init__(self) -> None:
        self._probe_cache: dict[str, HFVLLMRuntimeProbe] = {}

    def get_cached(self, repo_id: str) -> HFVLLMRuntimeProbe | None:
        return self._probe_cache.get(repo_id)

    def set_cached(self, repo_id: str, probe: HFVLLMRuntimeProbe) -> HFVLLMRuntimeProbe:
        if probe.observed_at is None:
            probe.observed_at = datetime.now(UTC)
        self._probe_cache[repo_id] = probe
        return probe

    async def probe_runtime(
        self,
        repo_id: str,
        support: HFVLLMSupport,
    ) -> HFVLLMRuntimeProbe:
        cached = self.get_cached(repo_id)
        if cached is not None:
            return cached

        local_dir = self.resolve_local_model_dir(repo_id)
        if local_dir is None:
            return self.set_cached(
                repo_id,
                HFVLLMRuntimeProbe(
                    status="unavailable",
                    reason="No hay artefacto local o snapshot cacheado para ejecutar un probe real de vLLM.",
                    recommended_model_impl=support.model_impl,
                    recommended_runner=support.runner,
                ),
            )

        command = self._build_probe_command(repo_id, local_dir, support)
        try:
            process = await asyncio.create_subprocess_exec(
                *command.cmd,
                env=command.env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            return self.set_cached(
                repo_id,
                HFVLLMRuntimeProbe(
                    status="unavailable",
                    reason=f"No se pudo lanzar el probe real de vLLM: {exc}",
                    recommended_model_impl=support.model_impl,
                    recommended_runner=support.runner,
                ),
            )

        try:
            await self._wait_for_health(process, command.port)
            status = {
                "native_vllm": "supported_native",
                "transformers_backend": "supported_transformers_backend",
                "pooling": "supported_pooling",
            }.get(support.classification, "unknown")
            return self.set_cached(
                repo_id,
                HFVLLMRuntimeProbe(
                    status=status,  # type: ignore[arg-type]
                    recommended_model_impl=support.model_impl,
                    recommended_runner=support.runner,
                    tokenizer_load=True,
                    config_load=True,
                ),
            )
        except Exception:
            reason = await self._infer_failure_reason(process, support)
            return self.set_cached(repo_id, reason)
        finally:
            await self._terminate_process(process)

    def resolve_local_model_dir(self, repo_id: str) -> Path | None:
        base = Path(settings.models_dir)
        direct = base / repo_id
        if direct.exists() and direct.is_dir():
            return direct

        hf_layout = base / "huggingface" / repo_id.replace("/", "--")
        if hf_layout.exists() and hf_layout.is_dir():
            return hf_layout

        cached = self._resolve_hf_cache_snapshot_dir(repo_id)
        if cached is not None:
            return cached
        return None

    def _resolve_hf_cache_snapshot_dir(self, repo_id: str) -> Path | None:
        hf_cache_dir = getattr(settings, "hf_cache_dir", "")
        if not hf_cache_dir:
            return None

        model_cache_root = Path(hf_cache_dir) / "hub" / f"models--{repo_id.replace('/', '--')}"
        snapshots_dir = model_cache_root / "snapshots"
        if not snapshots_dir.exists() or not snapshots_dir.is_dir():
            return None

        ref_main = model_cache_root / "refs" / "main"
        if ref_main.exists():
            try:
                commit = ref_main.read_text(encoding="utf-8").strip()
            except Exception:
                commit = ""
            if commit:
                snap = snapshots_dir / commit
                if snap.exists() and snap.is_dir():
                    return snap

        candidates = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _build_probe_command(
        self,
        repo_id: str,
        model_dir: Path,
        support: HFVLLMSupport,
    ) -> _ProbeCommand:
        port = self._reserve_free_port()
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            repo_id,
            "--gpu-memory-utilization",
            str(settings.vllm_gpu_memory_utilization),
        ]
        if settings.vllm_enforce_eager:
            cmd.append("--enforce-eager")
        if support.model_impl:
            cmd.extend(["--model-impl", support.model_impl])
        if support.runner:
            cmd.extend(["--runner", support.runner])

        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": settings.cuda_device_order,
            "CUDA_VISIBLE_DEVICES": str(settings.default_gpu_index),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HF_HOME": settings.hf_cache_dir,
        }
        if settings.hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

        return _ProbeCommand(cmd=cmd, env=env, target=str(model_dir), port=port)

    async def _wait_for_health(
        self, process: asyncio.subprocess.Process, port: int
    ) -> None:
        deadline = asyncio.get_running_loop().time() + _PROBE_STARTUP_TIMEOUT_S
        async with httpx.AsyncClient(timeout=2.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                if process.returncode is not None:
                    raise RuntimeError(f"Probe process exited early with code {process.returncode}")
                try:
                    response = await client.get(f"http://127.0.0.1:{port}/health")
                    if response.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        raise TimeoutError("vLLM probe healthcheck timed out")

    async def _infer_failure_reason(
        self,
        process: asyncio.subprocess.Process,
        support: HFVLLMSupport,
    ) -> HFVLLMRuntimeProbe:
        stderr = await self._read_stderr_tail(process)
        stderr_lower = stderr.lower()
        if "trust_remote_code=true" in stderr_lower or "remote code" in stderr_lower:
            return HFVLLMRuntimeProbe(
                status="needs_remote_code",
                reason="El probe real de vLLM requiere trust_remote_code.",
                recommended_model_impl=support.model_impl,
                recommended_runner=support.runner,
                config_load=False,
                tokenizer_load=False,
            )
        if self._looks_like_missing_chat_template(stderr_lower):
            return HFVLLMRuntimeProbe(
                status="missing_chat_template",
                reason=(
                    stderr
                    or "El probe real de vLLM indica que falta un chat_template compatible."
                ),
                recommended_model_impl=support.model_impl,
                recommended_runner=support.runner,
                config_load=True,
                tokenizer_load=True,
            )
        if self._looks_like_missing_tool_parser(stderr_lower):
            return HFVLLMRuntimeProbe(
                status="missing_tool_parser",
                reason=(
                    stderr
                    or "El probe real de vLLM indica que falta configurar tool_call_parser."
                ),
                recommended_model_impl=support.model_impl,
                recommended_runner=support.runner,
                config_load=True,
                tokenizer_load=True,
            )
        if self._looks_like_missing_reasoning_parser(stderr_lower):
            return HFVLLMRuntimeProbe(
                status="missing_reasoning_parser",
                reason=(
                    stderr
                    or "El probe real de vLLM indica que falta configurar reasoning_parser."
                ),
                recommended_model_impl=support.model_impl,
                recommended_runner=support.runner,
                config_load=True,
                tokenizer_load=True,
            )
        if self._looks_like_needing_hf_overrides(stderr_lower):
            return HFVLLMRuntimeProbe(
                status="needs_hf_overrides",
                reason=(
                    stderr
                    or "El probe real de vLLM sugiere que el modelo necesita hf_overrides."
                ),
                recommended_model_impl=support.model_impl,
                recommended_runner=support.runner,
                config_load=True,
                tokenizer_load=True,
            )
        if "tokenizer" in stderr_lower and "not supported" in stderr_lower:
            return HFVLLMRuntimeProbe(
                status="unsupported_tokenizer",
                reason=stderr or "El probe real de vLLM fallo por tokenizer no soportado.",
                recommended_model_impl=support.model_impl,
                recommended_runner=support.runner,
                config_load=True,
                tokenizer_load=False,
            )
        if "no module named" in stderr_lower and "vllm" in stderr_lower:
            return HFVLLMRuntimeProbe(
                status="unavailable",
                reason="vLLM no esta disponible en este entorno para ejecutar el probe real.",
                recommended_model_impl=support.model_impl,
                recommended_runner=support.runner,
            )
        return HFVLLMRuntimeProbe(
            status="unsupported_architecture" if support.classification == "unknown" else "unavailable",
            reason=stderr or "El probe real de vLLM no pudo arrancar correctamente.",
            recommended_model_impl=support.model_impl,
            recommended_runner=support.runner,
            config_load=None,
            tokenizer_load=None,
        )

    def _looks_like_missing_chat_template(self, stderr_lower: str) -> bool:
        patterns = (
            "chat template is not set",
            "chat_template is not set",
            "cannot use apply_chat_template",
            "default chat template is no longer allowed",
            "missing chat template",
            "requires a chat template",
        )
        return any(pattern in stderr_lower for pattern in patterns)

    def _looks_like_missing_tool_parser(self, stderr_lower: str) -> bool:
        patterns = (
            "tool call parser",
            "tool_call_parser",
            "--tool-call-parser",
            "tool parser plugin",
            "auto tool choice",
        )
        return any(pattern in stderr_lower for pattern in patterns)

    def _looks_like_missing_reasoning_parser(self, stderr_lower: str) -> bool:
        patterns = (
            "reasoning parser",
            "reasoning_parser",
            "--reasoning-parser",
            "enable reasoning",
        )
        return any(pattern in stderr_lower for pattern in patterns)

    def _looks_like_needing_hf_overrides(self, stderr_lower: str) -> bool:
        patterns = (
            "hf_overrides",
            "hf overrides",
            "rope_scaling",
            "mrope",
            "mrope_section",
            "sliding_window_pattern",
        )
        return any(pattern in stderr_lower for pattern in patterns)

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=_PROBE_SHUTDOWN_TIMEOUT_S)
        except Exception:
            try:
                process.kill()
                await process.wait()
            except Exception:
                return

    async def _read_stderr_tail(
        self, process: asyncio.subprocess.Process, limit: int = 4000
    ) -> str:
        if process.stderr is None:
            return ""
        try:
            stderr_data = await asyncio.wait_for(process.stderr.read(), timeout=0.5)
        except Exception:
            return ""
        if not stderr_data:
            return ""
        text = stderr_data.decode("utf-8", errors="replace").strip()
        if len(text) <= limit:
            return text
        head = text[: limit // 2]
        tail = text[-(limit // 2) :]
        return f"{head}\n...[truncated]...\n{tail}"

    def _reserve_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])
