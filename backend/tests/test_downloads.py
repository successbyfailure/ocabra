from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocabra.api.internal.downloads import DownloadManager
from ocabra.schemas.registry import DownloadJob


@pytest.mark.asyncio
async def test_auto_register_model_uses_register_config() -> None:
    manager = DownloadManager()
    model_manager = SimpleNamespace(
        get_state=AsyncMock(return_value=None),
        add_model=AsyncMock(),
    )
    manager._app = SimpleNamespace(state=SimpleNamespace(model_manager=model_manager))
    manager._hf_registry.infer_backend_for_repo = AsyncMock(return_value="vllm")

    job = DownloadJob(
        job_id="job-1",
        source="huggingface",
        model_ref="Qwen/Qwen3-8B-Instruct",
        artifact=None,
        register_config={
            "display_name": "Qwen3 8B",
            "load_policy": "warm",
            "extra_config": {
                "vllm": {
                    "model_impl": "vllm",
                    "runner": "generate",
                    "tool_call_parser": "qwen3_json",
                }
            },
        },
        status="completed",
        progress_pct=100.0,
        speed_mb_s=None,
        eta_seconds=0,
        error=None,
        started_at="2026-03-14T00:00:00Z",
        completed_at="2026-03-14T00:01:00Z",
    )

    await manager._auto_register_model(job)

    model_manager.add_model.assert_awaited_once_with(
        model_id="Qwen/Qwen3-8B-Instruct",
        backend_type="vllm",
        display_name="Qwen3 8B",
        load_policy="warm",
        auto_reload=False,
        preferred_gpu=None,
        extra_config={
            "vllm": {
                "model_impl": "vllm",
                "runner": "generate",
                "tool_call_parser": "qwen3_json",
            }
        },
    )
