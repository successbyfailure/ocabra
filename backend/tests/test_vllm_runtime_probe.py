from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.schemas.registry import HFVLLMSupport


def _fake_proc(returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    return proc


@pytest.mark.asyncio
async def test_probe_runtime_succeeds_for_local_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "org" / "demo-model"
    model_dir.mkdir(parents=True)
    proc = _fake_proc(returncode=None)

    from ocabra.registry.vllm_runtime_probe import VLLMRuntimeProbeService

    with (
        patch("ocabra.registry.vllm_runtime_probe.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        patch.object(VLLMRuntimeProbeService, "_wait_for_health", new=AsyncMock()),
    ):
        mock_settings.models_dir = str(tmp_path / "models")
        mock_settings.hf_cache_dir = str(tmp_path / "hf_cache")
        mock_settings.default_gpu_index = 1
        mock_settings.cuda_device_order = "PCI_BUS_ID"
        mock_settings.hf_token = ""
        mock_settings.vllm_gpu_memory_utilization = 0.85
        mock_settings.vllm_enforce_eager = True

        service = VLLMRuntimeProbeService()
        probe = await service.probe_runtime(
            "org/demo-model",
            HFVLLMSupport(classification="native_vllm", model_impl="vllm", runner="generate"),
        )

    assert probe.status == "supported_native"
    assert probe.config_load is True
    assert probe.tokenizer_load is True


@pytest.mark.asyncio
async def test_probe_runtime_returns_unavailable_without_local_artifact(
    tmp_path: Path,
) -> None:
    from ocabra.registry.vllm_runtime_probe import VLLMRuntimeProbeService

    with patch("ocabra.registry.vllm_runtime_probe.settings") as mock_settings:
        mock_settings.models_dir = str(tmp_path / "models")
        mock_settings.hf_cache_dir = str(tmp_path / "hf_cache")
        service = VLLMRuntimeProbeService()
        probe = await service.probe_runtime(
            "org/missing-model",
            HFVLLMSupport(
                classification="transformers_backend",
                model_impl="transformers",
                runner="generate",
            ),
        )

    assert probe.status == "unavailable"
    assert "artefacto local" in (probe.reason or "")


@pytest.mark.asyncio
async def test_probe_runtime_detects_remote_code_requirement(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "org" / "demo-model"
    model_dir.mkdir(parents=True)
    proc = _fake_proc(returncode=1)
    proc.stderr.read = AsyncMock(return_value=b"Please pass trust_remote_code=True to load this model")

    from ocabra.registry.vllm_runtime_probe import VLLMRuntimeProbeService

    with (
        patch("ocabra.registry.vllm_runtime_probe.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        patch.object(
            VLLMRuntimeProbeService,
            "_wait_for_health",
            new=AsyncMock(side_effect=RuntimeError("startup failed")),
        ),
    ):
        mock_settings.models_dir = str(tmp_path / "models")
        mock_settings.hf_cache_dir = str(tmp_path / "hf_cache")
        mock_settings.default_gpu_index = 1
        mock_settings.cuda_device_order = "PCI_BUS_ID"
        mock_settings.hf_token = ""
        mock_settings.vllm_gpu_memory_utilization = 0.85
        mock_settings.vllm_enforce_eager = True

        service = VLLMRuntimeProbeService()
        probe = await service.probe_runtime(
            "org/demo-model",
            HFVLLMSupport(
                classification="transformers_backend",
                model_impl="transformers",
                runner="generate",
            ),
        )

    assert probe.status == "needs_remote_code"
    assert "trust_remote_code" in (probe.reason or "")


@pytest.mark.asyncio
async def test_probe_runtime_detects_missing_chat_template(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "org" / "chat-model"
    model_dir.mkdir(parents=True)
    proc = _fake_proc(returncode=1)
    proc.stderr.read = AsyncMock(
        return_value=b"As of transformers v4.44, default chat template is no longer allowed."
    )

    from ocabra.registry.vllm_runtime_probe import VLLMRuntimeProbeService

    with (
        patch("ocabra.registry.vllm_runtime_probe.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        patch.object(
            VLLMRuntimeProbeService,
            "_wait_for_health",
            new=AsyncMock(side_effect=RuntimeError("startup failed")),
        ),
    ):
        mock_settings.models_dir = str(tmp_path / "models")
        mock_settings.hf_cache_dir = str(tmp_path / "hf_cache")
        mock_settings.default_gpu_index = 0
        mock_settings.cuda_device_order = "PCI_BUS_ID"
        mock_settings.hf_token = ""
        mock_settings.vllm_gpu_memory_utilization = 0.85
        mock_settings.vllm_enforce_eager = True

        service = VLLMRuntimeProbeService()
        probe = await service.probe_runtime(
            "org/chat-model",
            HFVLLMSupport(classification="native_vllm", model_impl="vllm", runner="generate"),
        )

    assert probe.status == "missing_chat_template"
    assert probe.config_load is True
    assert probe.tokenizer_load is True


@pytest.mark.asyncio
async def test_probe_runtime_detects_missing_tool_parser(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "org" / "tool-model"
    model_dir.mkdir(parents=True)
    proc = _fake_proc(returncode=1)
    proc.stderr.read = AsyncMock(
        return_value=b"Automatic tool choice requires --tool-call-parser to be set."
    )

    from ocabra.registry.vllm_runtime_probe import VLLMRuntimeProbeService

    with (
        patch("ocabra.registry.vllm_runtime_probe.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        patch.object(
            VLLMRuntimeProbeService,
            "_wait_for_health",
            new=AsyncMock(side_effect=RuntimeError("startup failed")),
        ),
    ):
        mock_settings.models_dir = str(tmp_path / "models")
        mock_settings.hf_cache_dir = str(tmp_path / "hf_cache")
        mock_settings.default_gpu_index = 0
        mock_settings.cuda_device_order = "PCI_BUS_ID"
        mock_settings.hf_token = ""
        mock_settings.vllm_gpu_memory_utilization = 0.85
        mock_settings.vllm_enforce_eager = True

        service = VLLMRuntimeProbeService()
        probe = await service.probe_runtime(
            "org/tool-model",
            HFVLLMSupport(classification="native_vllm", model_impl="vllm", runner="generate"),
        )

    assert probe.status == "missing_tool_parser"


@pytest.mark.asyncio
async def test_probe_runtime_detects_missing_reasoning_parser(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "org" / "reasoning-model"
    model_dir.mkdir(parents=True)
    proc = _fake_proc(returncode=1)
    proc.stderr.read = AsyncMock(
        return_value=b"Reasoning is enabled but no --reasoning-parser was provided."
    )

    from ocabra.registry.vllm_runtime_probe import VLLMRuntimeProbeService

    with (
        patch("ocabra.registry.vllm_runtime_probe.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        patch.object(
            VLLMRuntimeProbeService,
            "_wait_for_health",
            new=AsyncMock(side_effect=RuntimeError("startup failed")),
        ),
    ):
        mock_settings.models_dir = str(tmp_path / "models")
        mock_settings.hf_cache_dir = str(tmp_path / "hf_cache")
        mock_settings.default_gpu_index = 0
        mock_settings.cuda_device_order = "PCI_BUS_ID"
        mock_settings.hf_token = ""
        mock_settings.vllm_gpu_memory_utilization = 0.85
        mock_settings.vllm_enforce_eager = True

        service = VLLMRuntimeProbeService()
        probe = await service.probe_runtime(
            "org/reasoning-model",
            HFVLLMSupport(classification="native_vllm", model_impl="vllm", runner="generate"),
        )

    assert probe.status == "missing_reasoning_parser"


@pytest.mark.asyncio
async def test_probe_runtime_detects_need_for_hf_overrides(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "org" / "override-model"
    model_dir.mkdir(parents=True)
    proc = _fake_proc(returncode=1)
    proc.stderr.read = AsyncMock(
        return_value=b"KeyError: rope_scaling contains unsupported key mrope_section"
    )

    from ocabra.registry.vllm_runtime_probe import VLLMRuntimeProbeService

    with (
        patch("ocabra.registry.vllm_runtime_probe.settings") as mock_settings,
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        patch.object(
            VLLMRuntimeProbeService,
            "_wait_for_health",
            new=AsyncMock(side_effect=RuntimeError("startup failed")),
        ),
    ):
        mock_settings.models_dir = str(tmp_path / "models")
        mock_settings.hf_cache_dir = str(tmp_path / "hf_cache")
        mock_settings.default_gpu_index = 0
        mock_settings.cuda_device_order = "PCI_BUS_ID"
        mock_settings.hf_token = ""
        mock_settings.vllm_gpu_memory_utilization = 0.85
        mock_settings.vllm_enforce_eager = True

        service = VLLMRuntimeProbeService()
        probe = await service.probe_runtime(
            "org/override-model",
            HFVLLMSupport(classification="native_vllm", model_impl="vllm", runner="generate"),
        )

    assert probe.status == "needs_hf_overrides"
