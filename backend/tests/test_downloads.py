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
        pin_permanent=False,
        preferred_gpu=None,
        extra_config={
            "vllm": {
                "model_impl": "vllm",
                "runner": "generate",
                "tool_call_parser": "qwen3_json",
            }
        },
        model_key="Qwen/Qwen3-8B-Instruct",
        backend_model_id_override="Qwen/Qwen3-8B-Instruct",
        artifact=None,
    )


@pytest.mark.asyncio
async def test_update_job_does_not_auto_register_new_deployments() -> None:
    manager = DownloadManager()
    manager._ollama_registry.pull = AsyncMock()
    manager._save_and_publish = AsyncMock()
    manager._auto_register_model = AsyncMock()
    manager._refresh_registry_metadata = AsyncMock()
    job = DownloadJob(
        job_id="job-update-1",
        source="ollama",
        model_ref="gemma4:26b",
        artifact=None,
        register_config=None,
        status="queued",
        progress_pct=0.0,
        speed_mb_s=None,
        eta_seconds=None,
        error=None,
        started_at="2026-11-09T00:00:00Z",
        completed_at=None,
        kind="update",
    )

    await manager._execute_job(job)

    manager._ollama_registry.pull.assert_awaited_once()
    manager._auto_register_model.assert_not_awaited()
    manager._refresh_registry_metadata.assert_awaited_once()


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
    assert call["extra_config"]["bitnet"]["model_path"].endswith(
        "/huggingface/microsoft--BitNet-b1.58-2B-4T-gguf--ggml-model-i2_s/ggml-model-i2_s.gguf"
    )
    assert "model_path" not in call["extra_config"]


def _gguf_job(register_config: dict) -> DownloadJob:
    return DownloadJob(
        job_id="job-gguf-1",
        source="huggingface",
        model_ref="unsloth/Qwen3-8B-GGUF",
        artifact="Qwen3-8B-Q4_K_M.gguf",
        register_config=register_config,
        status="completed",
        progress_pct=100.0,
        speed_mb_s=None,
        eta_seconds=0,
        error=None,
        started_at="2026-03-14T00:00:00Z",
        completed_at="2026-03-14T00:01:00Z",
    )


@pytest.mark.asyncio
async def test_auto_register_gguf_pins_path_in_llama_cpp_section_keeping_user_keys() -> None:
    manager = DownloadManager()
    model_manager = SimpleNamespace(get_state=AsyncMock(return_value=None), add_model=AsyncMock())
    manager._app = SimpleNamespace(state=SimpleNamespace(model_manager=model_manager))
    manager._hf_registry.infer_backend_for_repo = AsyncMock(return_value="llama_cpp")

    job = _gguf_job(
        {"backend_type": "llamacpp_mks", "extra_config": {"llama_cpp": {"ctx_size": 8192}}}
    )
    await manager._auto_register_model(job)

    call = model_manager.add_model.await_args.kwargs
    assert call["model_id"] == "llamacpp_mks/unsloth/Qwen3-8B-GGUF::Qwen3-8B-Q4_K_M"
    section = call["extra_config"]["llama_cpp"]
    assert section["ctx_size"] == 8192
    assert section["model_path"].endswith(
        "/unsloth--Qwen3-8B-GGUF--Qwen3-8B-Q4_K_M/Qwen3-8B-Q4_K_M.gguf"
    )
    assert "model_path" not in call["extra_config"]


@pytest.mark.asyncio
async def test_registry_metadata_refresh_targets_the_backend_the_user_picked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the refresh re-inferred the backend (``llama_cpp``) and wrote
    the metadata to a model id that was never registered when the user had
    picked ``llamacpp_mks`` in Explore."""
    manager = DownloadManager()
    model_manager = SimpleNamespace(set_registry_metadata=AsyncMock())
    manager._app = SimpleNamespace(state=SimpleNamespace(model_manager=model_manager))
    manager._hf_registry.infer_backend_for_repo = AsyncMock(return_value="llama_cpp")
    fetch = AsyncMock(return_value={"source": "huggingface"})
    monkeypatch.setattr("ocabra.registry.metadata.fetch_registry_metadata", fetch)

    await manager._refresh_registry_metadata(_gguf_job({"backend_type": "llamacpp_mks"}))

    model_manager.set_registry_metadata.assert_awaited_once_with(
        "llamacpp_mks/unsloth/Qwen3-8B-GGUF::Qwen3-8B-Q4_K_M", {"source": "huggingface"}
    )
    manager._hf_registry.infer_backend_for_repo.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_register_hf_nemo_sets_base_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        _job(
            "old-failed", "failed", "2026-10-25T00:00:00Z"
        ),  # past the (longer) failed retention too
        _job(
            "recent-failed", "failed", "2026-11-01T12:00:00Z"
        ),  # past completed's window, within failed's
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


@pytest.mark.asyncio
async def test_ollama_pull_is_served_by_llamacpp_mks_and_keeps_ollama_as_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama is the distribution channel; the production llama.cpp serves it
    (reading the tag's blob through its manifest) and the Ollama deployment
    stays as a second, non-default deployment of the same model."""
    from ocabra.core import model_deployments as md

    manager = DownloadManager()
    model_manager = SimpleNamespace(get_state=AsyncMock(return_value=None), add_model=AsyncMock())
    profile_registry = SimpleNamespace(on_model_added=AsyncMock(return_value="gemma4:e4b"))
    installer_states = [
        SimpleNamespace(backend_type=b, install_status=md_status)
        for b, md_status in (("llamacpp_mks", "installed"), ("ollama", "installed"))
    ]
    manager._app = SimpleNamespace(
        state=SimpleNamespace(
            model_manager=model_manager,
            profile_registry=profile_registry,
            backend_installer=SimpleNamespace(list_states=lambda: installer_states),
        )
    )
    monkeypatch.setattr(md, "installed_backends", lambda _state: {"llamacpp_mks", "ollama"})
    add_deployment = AsyncMock()
    monkeypatch.setattr(md, "add_deployment", add_deployment)

    job = DownloadJob(
        job_id="job-ollama-1",
        source="ollama",
        model_ref="gemma4:e4b",
        artifact=None,
        register_config={},
        status="completed",
        progress_pct=100.0,
        speed_mb_s=None,
        eta_seconds=0,
        error=None,
        started_at="2026-11-07T00:00:00Z",
        completed_at="2026-11-07T00:01:00Z",
    )
    await manager._auto_register_model(job)

    call = model_manager.add_model.await_args.kwargs
    assert call["model_id"] == "llamacpp_mks/gemma4:e4b"
    assert call["model_key"] == "gemma4:e4b"
    assert call["artifact"]["source"] == "ollama" and call["artifact"]["ref"] == "gemma4:e4b"
    add_deployment.assert_awaited_once()
    assert add_deployment.await_args.args == ("gemma4:e4b", "ollama")


@pytest.mark.asyncio
async def test_un_pull_de_ollama_servido_por_llama_cpp_se_registra_usable() -> None:
    """Sin esto el alta deja extra_config vacio y llama.cpp arranca en CPU.

    Instalar desde Explore no manda ``extra_config``, y desde 0.47.0 un pull
    de Ollama registra tambien un despliegue ``llamacpp_mks``. Con la seccion
    vacia, ``_build_options`` cae a ``LLAMA_CPP_GPU_LAYERS=0`` y
    ``LLAMA_CPP_CTX_SIZE=4096``: un modelo de 20 GiB intenta correr en CPU y
    no carga. Este test mira lo que llega a ``add_model``, no la funcion de
    defaults por separado.
    """
    manager = DownloadManager()
    model_manager = SimpleNamespace(get_state=AsyncMock(return_value=None), add_model=AsyncMock())
    manager._app = SimpleNamespace(state=SimpleNamespace(model_manager=model_manager))
    manager._resolve_job_backend = AsyncMock(return_value="llamacpp_mks")

    job = DownloadJob(
        job_id="job-ollama",
        source="ollama",
        model_ref="ornith-1.5:35b",
        artifact=None,
        register_config={},
        status="completed",
        progress_pct=100.0,
        speed_mb_s=None,
        eta_seconds=0,
        error=None,
        started_at="2026-11-09T00:04:00Z",
        completed_at="2026-11-09T00:05:00Z",
    )

    await manager._auto_register_model(job)

    enviado = model_manager.add_model.await_args.kwargs["extra_config"]
    seccion = enviado["llama_cpp"]
    assert seccion["gpu_layers"] == 99
    assert seccion["ctx_size"] > 4096
    assert seccion["flash_attn"] is True
    assert seccion["cache_type_k"] == "q8_0"
    assert seccion["parallel_slots"] == 4


@pytest.mark.asyncio
async def test_a_un_backend_que_no_es_llama_cpp_no_se_le_tocan_los_defaults() -> None:
    manager = DownloadManager()
    model_manager = SimpleNamespace(get_state=AsyncMock(return_value=None), add_model=AsyncMock())
    manager._app = SimpleNamespace(state=SimpleNamespace(model_manager=model_manager))
    manager._resolve_job_backend = AsyncMock(return_value="ollama")

    job = DownloadJob(
        job_id="job-ollama-2",
        source="ollama",
        model_ref="ornith-1.5:35b",
        artifact=None,
        register_config={},
        status="completed",
        progress_pct=100.0,
        speed_mb_s=None,
        eta_seconds=0,
        error=None,
        started_at="2026-11-09T00:04:00Z",
        completed_at="2026-11-09T00:05:00Z",
    )

    await manager._auto_register_model(job)

    assert model_manager.add_model.await_args.kwargs["extra_config"] is None
