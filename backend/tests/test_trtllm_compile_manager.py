"""Tests for TrtllmCompileManager — mock Docker, full convert→build cycle."""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.core.trtllm_compile_manager import (
    CompileJobState,
    CompileRequest,
    TrtllmCompileManager,
)


# ── helpers ──────────────────────────────────────────────────────


def _make_req(
    source_model: str = "vllm/Qwen/Qwen3.5-27B-GPTQ-Int4",
    engine_name: str = "Qwen3.5-27B-fp16",
    gpu_indices: list[int] | None = None,
    dtype: str = "fp16",
    max_batch_size: int = 1,
    max_input_len: int = 2048,
    max_seq_len: int = 4096,
) -> CompileRequest:
    return CompileRequest(
        source_model=source_model,
        engine_name=engine_name,
        gpu_indices=gpu_indices or [1],
        dtype=dtype,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_seq_len=max_seq_len,
    )


def _make_proc(returncode: int = 0, stdout_lines: list[bytes] | None = None) -> MagicMock:
    """Return a mock asyncio.subprocess.Process."""
    proc = MagicMock()
    proc.returncode = returncode

    lines = stdout_lines or []

    async def _aiter_lines():
        for line in lines:
            yield line

    stdout_mock = MagicMock()
    stdout_mock.__aiter__ = lambda self: _aiter_lines()
    proc.stdout = stdout_mock

    async def _wait():
        return returncode

    proc.wait = _wait
    proc.kill = MagicMock()
    return proc


def _patch_db(manager: TrtllmCompileManager) -> None:
    """Replace DB methods with no-ops."""
    manager._save_to_db = AsyncMock()
    manager._load_from_db = AsyncMock(return_value=None)
    manager._recover_stale_jobs = AsyncMock()


def _patch_redis(manager: TrtllmCompileManager) -> None:
    """Replace Redis publish methods with no-ops."""
    manager._publish_progress = AsyncMock()
    manager._publish_log_line = AsyncMock()


# ── tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_returns_pending_state() -> None:
    manager = TrtllmCompileManager()
    _patch_db(manager)
    _patch_redis(manager)

    req = _make_req()
    state = await manager.enqueue(req)

    assert state.status == "pending"
    assert state.source_model == req.source_model
    assert state.engine_name == req.engine_name
    assert state.gpu_indices == [1]
    assert state.job_id in manager._history


@pytest.mark.asyncio
async def test_successful_compile_cycle() -> None:
    """Full convert→build cycle with mock Docker — job ends as 'done'."""
    manager = TrtllmCompileManager()
    _patch_db(manager)
    _patch_redis(manager)
    await manager._recover_stale_jobs()  # already mocked

    convert_proc = _make_proc(returncode=0, stdout_lines=[b"[convert] done\n"])
    build_proc = _make_proc(returncode=0, stdout_lines=[b"[build] done\n"])
    procs = iter([convert_proc, build_proc])

    async def _fake_subprocess(*args, **kwargs):
        return next(procs)

    with patch(
        "ocabra.core.trtllm_compile_manager.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess,
    ):
        req = _make_req()
        state = await manager.enqueue(req)

        # Start worker and let it process the job
        manager._worker_task = asyncio.create_task(manager._worker_loop())
        # Give the worker time to process
        await asyncio.sleep(0.1)
        manager._worker_task.cancel()
        try:
            await manager._worker_task
        except asyncio.CancelledError:
            pass

    assert state.status == "done"
    assert state.progress_pct == 100
    assert state.engine_dir is not None
    assert "Qwen3.5-27B-fp16" in state.engine_dir


@pytest.mark.asyncio
async def test_failed_compile_on_docker_error() -> None:
    """Job transitions to 'failed' when Docker exits with non-zero code."""
    manager = TrtllmCompileManager()
    _patch_db(manager)
    _patch_redis(manager)

    fail_proc = _make_proc(
        returncode=1,
        stdout_lines=[b"ERROR: conversion failed\n"],
    )

    async def _fake_subprocess(*args, **kwargs):
        return fail_proc

    with patch(
        "ocabra.core.trtllm_compile_manager.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess,
    ):
        req = _make_req()
        state = await manager.enqueue(req)

        manager._worker_task = asyncio.create_task(manager._worker_loop())
        await asyncio.sleep(0.1)
        manager._worker_task.cancel()
        try:
            await manager._worker_task
        except asyncio.CancelledError:
            pass

    assert state.status == "failed"
    assert state.error_detail is not None
    assert "1" in state.error_detail  # rc=1


@pytest.mark.asyncio
async def test_cancel_pending_job() -> None:
    """Cancelling a pending job sets status='cancelled' without running Docker."""
    manager = TrtllmCompileManager()
    _patch_db(manager)
    _patch_redis(manager)

    req = _make_req()
    state = await manager.enqueue(req)
    assert state.status == "pending"

    cancelled = await manager.cancel(state.job_id)
    assert cancelled.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_already_terminal_raises() -> None:
    """Cancelling a done job raises ValueError."""
    manager = TrtllmCompileManager()
    _patch_db(manager)
    _patch_redis(manager)

    job_id = uuid.uuid4().hex
    state = CompileJobState(
        job_id=job_id,
        source_model="vllm/test",
        engine_name="test-engine",
        gpu_indices=[1],
        dtype="fp16",
        config={},
        status="done",
    )
    manager._history[job_id] = state

    with pytest.raises(ValueError, match="terminal"):
        await manager.cancel(job_id)


@pytest.mark.asyncio
async def test_cancel_unknown_job_raises() -> None:
    """Cancelling an unknown job_id raises KeyError."""
    manager = TrtllmCompileManager()
    _patch_db(manager)
    _patch_redis(manager)

    with pytest.raises(KeyError, match="not found"):
        await manager.cancel("nonexistent-job-id")


@pytest.mark.asyncio
async def test_docker_cmd_single_gpu() -> None:
    """Check convert/build Docker commands for single GPU."""
    manager = TrtllmCompileManager()

    state = CompileJobState(
        job_id="abc",
        source_model="vllm/Qwen/Qwen3.5-27B-GPTQ-Int4",
        engine_name="Qwen3.5-27B-fp16",
        gpu_indices=[1],
        dtype="fp16",
        config={"max_batch_size": 1, "max_input_len": 2048, "max_seq_len": 4096},
    )

    with patch("ocabra.core.trtllm_compile_manager.settings") as mock_settings:
        mock_settings.tensorrt_llm_docker_bin = "docker"
        mock_settings.tensorrt_llm_docker_image = "nvcr.io/nvidia/tensorrt-llm/release:latest"
        mock_settings.tensorrt_llm_docker_models_mount_host = "/docker/ai-models/ocabra/models"
        mock_settings.tensorrt_llm_docker_models_mount_container = "/data/models"

        convert_cmd = manager._build_docker_cmd(state, "convert")
        build_cmd = manager._build_docker_cmd(state, "build")

    # GPU device
    assert "device=1" in " ".join(convert_cmd)
    assert "device=1" in " ".join(build_cmd)

    # Convert uses python3 -c (TRT-LLM 1.0, no trtllm-convert binary)
    assert "python3" in convert_cmd
    assert "-c" in convert_cmd

    # TP size passed as arg to convert script
    assert "--tp_size" in convert_cmd
    tp_idx = convert_cmd.index("--tp_size")
    assert convert_cmd[tp_idx + 1] == "1"

    # dtype mapped correctly: fp16 → float16
    assert "--dtype" in convert_cmd
    dtype_idx = convert_cmd.index("--dtype")
    assert convert_cmd[dtype_idx + 1] == "float16"

    # Source model path
    assert "Qwen--Qwen3.5-27B-GPTQ-Int4" in " ".join(convert_cmd)

    # Build uses trtllm-build
    assert "trtllm-build" in build_cmd
    assert "--max_batch_size" in build_cmd
    assert "--max_input_len" in build_cmd
    assert "--max_seq_len" in build_cmd


@pytest.mark.asyncio
async def test_docker_cmd_two_gpus() -> None:
    """TP=2 when two GPUs are requested."""
    manager = TrtllmCompileManager()

    state = CompileJobState(
        job_id="abc",
        source_model="vllm/Qwen/Qwen3.5-35B",
        engine_name="Qwen3.5-35B-tp2",
        gpu_indices=[0, 1],
        dtype="fp16",
        config={"max_batch_size": 1, "max_input_len": 2048, "max_seq_len": 4096},
    )

    with patch("ocabra.core.trtllm_compile_manager.settings") as mock_settings:
        mock_settings.tensorrt_llm_docker_bin = "docker"
        mock_settings.tensorrt_llm_docker_image = "nvcr.io/nvidia/tensorrt-llm/release:latest"
        mock_settings.tensorrt_llm_docker_models_mount_host = "/docker/ai-models/ocabra/models"
        mock_settings.tensorrt_llm_docker_models_mount_container = "/data/models"

        convert_cmd = manager._build_docker_cmd(state, "convert")

    assert "device=0,1" in " ".join(convert_cmd)
    assert "--tp_size" in convert_cmd
    tp_idx = convert_cmd.index("--tp_size")
    assert convert_cmd[tp_idx + 1] == "2"


@pytest.mark.asyncio
async def test_list_jobs_empty_returns_empty_list() -> None:
    """list_jobs returns empty list when DB has no rows."""
    manager = TrtllmCompileManager()

    async def _fake_list():
        return []

    manager.list_jobs = _fake_list  # type: ignore[method-assign]
    result = await manager.list_jobs()
    assert result == []
