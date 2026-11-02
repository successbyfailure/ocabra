from __future__ import annotations

from pathlib import Path
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
        model_id="vllm/Qwen/Qwen3-8B-Instruct",
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


@pytest.mark.asyncio
async def test_auto_register_bitnet_sets_model_path() -> None:
    manager = DownloadManager()
    model_manager = SimpleNamespace(
        get_state=AsyncMock(return_value=None),
        add_model=AsyncMock(),
    )
    manager._app = SimpleNamespace(state=SimpleNamespace(model_manager=model_manager))

    job = DownloadJob(
        job_id="job-bitnet-1",
        source="bitnet",
        model_ref="microsoft/BitNet-b1.58-2B-4T-gguf",
        artifact="ggml-model-i2_s.gguf",
        register_config={
            "display_name": "BitNet 2B",
            "load_policy": "on_demand",
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

    call = model_manager.add_model.await_args.kwargs
    assert call["backend_type"] == "bitnet"
    assert call["model_id"] == "bitnet/microsoft/BitNet-b1.58-2B-4T-gguf::ggml-model-i2_s"
    assert call["extra_config"]["model_path"].endswith(
        "/huggingface/microsoft--BitNet-b1.58-2B-4T-gguf--ggml-model-i2_s/ggml-model-i2_s.gguf"
    )


@pytest.mark.asyncio
async def test_auto_register_hf_nemo_sets_base_model_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ocabra.config import settings

    manager = DownloadManager()
    model_manager = SimpleNamespace(
        get_state=AsyncMock(return_value=None),
        add_model=AsyncMock(),
    )
    manager._app = SimpleNamespace(state=SimpleNamespace(model_manager=model_manager))
    manager._hf_registry.infer_backend_for_repo = AsyncMock(return_value="whisper")

    monkeypatch.setattr(settings, "models_dir", str(tmp_path), raising=False)
    download_dir = tmp_path / "huggingface" / "nvidia--canary-1b-v2"
    download_dir.mkdir(parents=True, exist_ok=True)
    nemo_file = download_dir / "canary-1b-v2.nemo"
    nemo_file.write_bytes(b"nemo")

    job = DownloadJob(
        job_id="job-nemo-1",
        source="huggingface",
        model_ref="nvidia/canary-1b-v2",
        artifact=None,
        register_config={
            "display_name": "Canary 1B v2",
            "load_policy": "on_demand",
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

    call = model_manager.add_model.await_args.kwargs
    assert call["backend_type"] == "whisper"
    assert call["extra_config"]["base_model_id"] == str(nemo_file)


def _job(job_id: str, status: str, completed_at: str | None) -> DownloadJob:
    return DownloadJob(
        job_id=job_id,
        source="ollama",
        model_ref="qwen3:8b",
        artifact=None,
        register_config=None,
        status=status,
        progress_pct=100.0 if status == "completed" else 0.0,
        speed_mb_s=None,
        eta_seconds=0,
        error="boom" if status == "failed" else None,
        started_at="2026-03-14T00:00:00Z",
        completed_at=completed_at,
    )


@pytest.mark.asyncio
async def test_sweep_old_jobs_deletes_past_retention_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """completed jobs sweep sooner than failed/cancelled ones (the error is
    the only record of what went wrong, worth keeping longer), and anything
    still active or younger than its retention window is left alone."""
    import ocabra.api.internal.downloads as downloads_mod

    manager = DownloadManager()
    jobs = [
        _job("old-completed", "completed", "2026-01-01T00:00:00Z"),  # past both cutoffs
        _job("recent-completed", "completed", "2026-11-01T23:59:00Z"),  # within completed retention
        _job("old-failed", "failed", "2026-10-25T00:00:00Z"),  # past the (longer) failed retention too
        _job("recent-failed", "failed", "2026-11-01T12:00:00Z"),  # past completed's window, within failed's
        _job("still-downloading", "downloading", None),  # not terminal — never swept
    ]
    manager.list_jobs = AsyncMock(return_value=jobs)
    fake_redis = SimpleNamespace(delete=AsyncMock())
    monkeypatch.setattr(downloads_mod, "get_redis_safe", lambda: fake_redis)

    class _FrozenDatetime:
        @staticmethod
        def now(_tz):
            from datetime import datetime as real_datetime

            return real_datetime(2026, 11, 2, 0, 0, 0, tzinfo=_tz)

    monkeypatch.setattr(downloads_mod, "datetime", _FrozenDatetime)

    deleted = await manager._sweep_old_jobs()

    assert deleted == 2
    fake_redis.delete.assert_any_await("download:job:old-completed")
    fake_redis.delete.assert_any_await("download:job:old-failed")
    assert fake_redis.delete.await_count == 2
