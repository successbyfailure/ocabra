"""TRT-LLM compile job mock tests.

Tests the compile manager lifecycle: enqueue, status tracking, convert->build phases,
cancellation, and error handling. All Docker subprocess calls and DB/Redis are mocked.
"""
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


# ── Helpers ──────────────────────────────────────────────────────


def _make_req(
    source_model: str = "vllm/TestOrg/TestModel-7B",
    engine_name: str = "TestModel-7B-fp16",
    gpu_indices: list[int] | None = None,
    dtype: str = "fp16",
) -> CompileRequest:
    return CompileRequest(
        source_model=source_model,
        engine_name=engine_name,
        gpu_indices=gpu_indices or [0],
        dtype=dtype,
        max_batch_size=1,
        max_input_len=2048,
        max_seq_len=4096,
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


def _setup_manager() -> TrtllmCompileManager:
    """Create a manager with DB and Redis mocked out."""
    manager = TrtllmCompileManager()
    manager._save_to_db = AsyncMock()
    manager._load_from_db = AsyncMock(return_value=None)
    manager._recover_stale_jobs = AsyncMock()
    manager._publish_progress = AsyncMock()
    manager._publish_log_line = AsyncMock()
    return manager


# ── Job creation and status tracking ────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_creates_pending_job():
    """enqueue() returns a job in 'pending' status with correct metadata."""
    manager = _setup_manager()
    req = _make_req()

    state = await manager.enqueue(req)

    assert state.status == "pending"
    assert state.source_model == req.source_model
    assert state.engine_name == req.engine_name
    assert state.gpu_indices == [0]
    assert state.dtype == "fp16"
    assert state.job_id in manager._history
    assert state.config == {
        "max_batch_size": 1,
        "max_input_len": 2048,
        "max_seq_len": 4096,
    }
    manager._save_to_db.assert_called()


@pytest.mark.asyncio
async def test_get_job_returns_cached_state():
    """get_job() returns the in-memory cached state."""
    manager = _setup_manager()
    req = _make_req()
    state = await manager.enqueue(req)

    retrieved = manager.get_job(state.job_id)
    assert retrieved is state


@pytest.mark.asyncio
async def test_get_job_unknown_returns_none():
    """get_job() returns None for unknown job IDs."""
    manager = _setup_manager()
    assert manager.get_job("nonexistent") is None


@pytest.mark.asyncio
async def test_to_dict_serialization():
    """CompileJobState.to_dict() produces a valid dict for API responses."""
    state = CompileJobState(
        job_id="abc123",
        source_model="vllm/Test/Model",
        engine_name="test-engine",
        gpu_indices=[0, 1],
        dtype="fp16",
        config={"max_batch_size": 4},
        status="running",
        phase="convert",
        progress_pct=25,
    )
    d = state.to_dict()
    assert d["job_id"] == "abc123"
    assert d["gpu_indices"] == [0, 1]
    assert d["status"] == "running"
    assert d["phase"] == "convert"
    assert d["progress_pct"] == 25
    assert d["started_at"] is None


# ── Convert -> Build phase flow ─────────────────────────────────


@pytest.mark.asyncio
async def test_successful_convert_build_cycle():
    """Full convert -> build cycle completes with status 'done'."""
    manager = _setup_manager()

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

        # Run the worker loop briefly
        task = asyncio.create_task(manager._worker_loop())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert state.status == "done"
    assert state.progress_pct == 100
    assert state.engine_dir is not None
    assert state.finished_at is not None
    assert state.error_detail is None


@pytest.mark.asyncio
async def test_phase_progression_convert_then_build():
    """Verify the phase changes from 'convert' to 'build' during execution."""
    manager = _setup_manager()
    observed_phases = []

    original_save = manager._save_to_db

    async def _capture_save(state):
        if state.phase:
            observed_phases.append(state.phase)

    manager._save_to_db = _capture_save

    convert_proc = _make_proc(returncode=0, stdout_lines=[b"ok\n"])
    build_proc = _make_proc(returncode=0, stdout_lines=[b"ok\n"])
    procs = iter([convert_proc, build_proc])

    async def _fake_subprocess(*args, **kwargs):
        return next(procs)

    with patch(
        "ocabra.core.trtllm_compile_manager.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess,
    ):
        req = _make_req()
        await manager.enqueue(req)

        task = asyncio.create_task(manager._worker_loop())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert "convert" in observed_phases
    assert "build" in observed_phases
    # convert comes before build
    assert observed_phases.index("convert") < observed_phases.index("build")


@pytest.mark.asyncio
async def test_progress_pct_advances_through_phases():
    """Progress % goes: 5 (convert start) -> 45 (convert done) -> 50 (build start) -> 95 (build done) -> 100."""
    manager = _setup_manager()
    observed_pcts = []

    async def _capture_save(state):
        observed_pcts.append(state.progress_pct)

    manager._save_to_db = _capture_save

    convert_proc = _make_proc(returncode=0, stdout_lines=[b"ok\n"])
    build_proc = _make_proc(returncode=0, stdout_lines=[b"ok\n"])
    procs = iter([convert_proc, build_proc])

    async def _fake_subprocess(*args, **kwargs):
        return next(procs)

    with patch(
        "ocabra.core.trtllm_compile_manager.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess,
    ):
        await manager.enqueue(_make_req())
        task = asyncio.create_task(manager._worker_loop())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Should contain the key progression points
    assert 5 in observed_pcts   # convert start
    assert 45 in observed_pcts  # convert done
    assert 50 in observed_pcts  # build start
    assert 95 in observed_pcts  # build done
    assert 100 in observed_pcts  # final


# ── Cancellation ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_pending_job():
    """Cancelling a pending job sets status='cancelled' immediately."""
    manager = _setup_manager()
    req = _make_req()
    state = await manager.enqueue(req)
    assert state.status == "pending"

    cancelled = await manager.cancel(state.job_id)
    assert cancelled.status == "cancelled"
    assert cancelled.finished_at is not None


@pytest.mark.asyncio
async def test_cancel_running_job_sets_event():
    """Cancelling a running job sets the cancel event; worker loop detects it."""
    manager = _setup_manager()

    # Create a slow process that yields lines slowly
    async def _aiter_slow():
        for i in range(100):
            await asyncio.sleep(0.05)
            yield f"line {i}\n".encode()

    slow_proc = MagicMock()
    slow_proc.returncode = 0
    stdout_mock = MagicMock()
    stdout_mock.__aiter__ = lambda self: _aiter_slow()
    slow_proc.stdout = stdout_mock
    slow_proc.kill = MagicMock()

    async def _wait():
        return 0

    slow_proc.wait = _wait

    async def _fake_subprocess(*args, **kwargs):
        return slow_proc

    with patch(
        "ocabra.core.trtllm_compile_manager.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess,
    ):
        req = _make_req()
        state = await manager.enqueue(req)

        task = asyncio.create_task(manager._worker_loop())
        await asyncio.sleep(0.05)  # Let it start running

        # Cancel while running
        cancelled = await manager.cancel(state.job_id)
        assert cancelled._cancel_event.is_set()

        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert state.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_already_done_raises():
    """Cancelling a job in a terminal state raises ValueError."""
    manager = _setup_manager()
    job_id = uuid.uuid4().hex
    state = CompileJobState(
        job_id=job_id,
        source_model="vllm/test",
        engine_name="test-engine",
        gpu_indices=[0],
        dtype="fp16",
        config={},
        status="done",
    )
    manager._history[job_id] = state

    with pytest.raises(ValueError, match="terminal"):
        await manager.cancel(job_id)


@pytest.mark.asyncio
async def test_cancel_already_failed_raises():
    """Cancelling a failed job raises ValueError."""
    manager = _setup_manager()
    job_id = uuid.uuid4().hex
    state = CompileJobState(
        job_id=job_id,
        source_model="vllm/test",
        engine_name="test-engine",
        gpu_indices=[0],
        dtype="fp16",
        config={},
        status="failed",
    )
    manager._history[job_id] = state

    with pytest.raises(ValueError, match="terminal"):
        await manager.cancel(job_id)


@pytest.mark.asyncio
async def test_cancel_unknown_job_raises():
    """Cancelling an unknown job raises KeyError."""
    manager = _setup_manager()
    with pytest.raises(KeyError, match="not found"):
        await manager.cancel("no-such-job")


@pytest.mark.asyncio
async def test_cancelled_job_skipped_by_worker():
    """A job cancelled while pending is skipped by the worker loop."""
    manager = _setup_manager()

    subprocess_called = False

    async def _fake_subprocess(*args, **kwargs):
        nonlocal subprocess_called
        subprocess_called = True
        return _make_proc()

    with patch(
        "ocabra.core.trtllm_compile_manager.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess,
    ):
        req = _make_req()
        state = await manager.enqueue(req)
        await manager.cancel(state.job_id)

        task = asyncio.create_task(manager._worker_loop())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert not subprocess_called


# ── Error handling ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_docker_convert_failure():
    """When Docker convert phase exits non-zero, job status becomes 'failed'."""
    manager = _setup_manager()

    fail_proc = _make_proc(
        returncode=1,
        stdout_lines=[b"ERROR: OOM during convert\n"],
    )

    async def _fake_subprocess(*args, **kwargs):
        return fail_proc

    with patch(
        "ocabra.core.trtllm_compile_manager.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess,
    ):
        req = _make_req()
        state = await manager.enqueue(req)

        task = asyncio.create_task(manager._worker_loop())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert state.status == "failed"
    assert state.error_detail is not None
    assert "1" in state.error_detail  # exit code


@pytest.mark.asyncio
async def test_docker_build_failure_after_successful_convert():
    """Convert succeeds but build fails -> job is 'failed'."""
    manager = _setup_manager()

    convert_proc = _make_proc(returncode=0, stdout_lines=[b"[convert] ok\n"])
    build_proc = _make_proc(returncode=2, stdout_lines=[b"ERROR: build OOM\n"])
    procs = iter([convert_proc, build_proc])

    async def _fake_subprocess(*args, **kwargs):
        return next(procs)

    with patch(
        "ocabra.core.trtllm_compile_manager.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess,
    ):
        req = _make_req()
        state = await manager.enqueue(req)

        task = asyncio.create_task(manager._worker_loop())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert state.status == "failed"
    assert "2" in state.error_detail  # exit code 2


@pytest.mark.asyncio
async def test_subprocess_exception_fails_job():
    """If subprocess creation itself raises, the job becomes 'failed'."""
    manager = _setup_manager()

    async def _exploding_subprocess(*args, **kwargs):
        raise OSError("Docker not found")

    with patch(
        "ocabra.core.trtllm_compile_manager.asyncio.create_subprocess_exec",
        side_effect=_exploding_subprocess,
    ):
        req = _make_req()
        state = await manager.enqueue(req)

        task = asyncio.create_task(manager._worker_loop())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert state.status == "failed"
    assert "Docker not found" in state.error_detail


# ── Docker command construction ─────────────────────────────────


@pytest.mark.asyncio
async def test_build_docker_cmd_convert_phase():
    """Convert phase uses python3 -c with the convert script."""
    manager = _setup_manager()
    state = CompileJobState(
        job_id="test",
        source_model="vllm/Meta/Llama-3-8B",
        engine_name="Llama-3-8B-fp16",
        gpu_indices=[0],
        dtype="fp16",
        config={"max_batch_size": 1, "max_input_len": 2048, "max_seq_len": 4096},
    )

    with patch("ocabra.core.trtllm_compile_manager.settings") as s:
        s.tensorrt_llm_docker_bin = "docker"
        s.tensorrt_llm_docker_image = "trtllm:latest"
        s.tensorrt_llm_docker_models_mount_host = "/models"
        s.tensorrt_llm_docker_models_mount_container = "/data/models"

        cmd = manager._build_docker_cmd(state, "convert")

    assert cmd[0] == "docker"
    assert "python3" in cmd
    assert "-c" in cmd
    assert "--dtype" in cmd
    dtype_idx = cmd.index("--dtype")
    assert cmd[dtype_idx + 1] == "float16"
    assert "--tp_size" in cmd
    assert "Meta--Llama-3-8B" in " ".join(cmd)


@pytest.mark.asyncio
async def test_build_docker_cmd_build_phase():
    """Build phase uses trtllm-build with correct args."""
    manager = _setup_manager()
    state = CompileJobState(
        job_id="test",
        source_model="vllm/Meta/Llama-3-8B",
        engine_name="Llama-3-8B-fp16",
        gpu_indices=[0],
        dtype="fp16",
        config={"max_batch_size": 2, "max_input_len": 1024, "max_seq_len": 2048},
    )

    with patch("ocabra.core.trtllm_compile_manager.settings") as s:
        s.tensorrt_llm_docker_bin = "docker"
        s.tensorrt_llm_docker_image = "trtllm:latest"
        s.tensorrt_llm_docker_models_mount_host = "/models"
        s.tensorrt_llm_docker_models_mount_container = "/data/models"

        cmd = manager._build_docker_cmd(state, "build")

    assert "trtllm-build" in cmd
    assert "--max_batch_size" in cmd
    assert cmd[cmd.index("--max_batch_size") + 1] == "2"
    assert cmd[cmd.index("--max_input_len") + 1] == "1024"
    assert cmd[cmd.index("--max_seq_len") + 1] == "2048"
    # Single GPU -> no --workers flag
    assert "--workers" not in cmd


@pytest.mark.asyncio
async def test_build_docker_cmd_multi_gpu_uses_workers():
    """TP>1 adds --workers flag to trtllm-build."""
    manager = _setup_manager()
    state = CompileJobState(
        job_id="test",
        source_model="vllm/Big/Model-70B",
        engine_name="Model-70B-tp2",
        gpu_indices=[0, 1],
        dtype="bf16",
        config={"max_batch_size": 1, "max_input_len": 2048, "max_seq_len": 4096},
    )

    with patch("ocabra.core.trtllm_compile_manager.settings") as s:
        s.tensorrt_llm_docker_bin = "docker"
        s.tensorrt_llm_docker_image = "trtllm:latest"
        s.tensorrt_llm_docker_models_mount_host = "/models"
        s.tensorrt_llm_docker_models_mount_container = "/data/models"

        cmd = manager._build_docker_cmd(state, "build")

    assert "--workers" in cmd
    assert cmd[cmd.index("--workers") + 1] == "2"
    # GPU spec for multi-GPU uses quoted format
    assert "device=0,1" in " ".join(cmd)


@pytest.mark.asyncio
async def test_build_docker_cmd_dtype_mapping():
    """dtype is mapped: bf16 -> bfloat16, fp16 -> float16."""
    manager = _setup_manager()

    for input_dtype, expected in [("fp16", "float16"), ("bf16", "bfloat16"), ("int8", "int8")]:
        state = CompileJobState(
            job_id="test",
            source_model="vllm/Org/Model",
            engine_name="model-test",
            gpu_indices=[0],
            dtype=input_dtype,
            config={"max_batch_size": 1, "max_input_len": 2048, "max_seq_len": 4096},
        )

        with patch("ocabra.core.trtllm_compile_manager.settings") as s:
            s.tensorrt_llm_docker_bin = "docker"
            s.tensorrt_llm_docker_image = "trtllm:latest"
            s.tensorrt_llm_docker_models_mount_host = "/models"
            s.tensorrt_llm_docker_models_mount_container = "/data/models"

            cmd = manager._build_docker_cmd(state, "convert")

        dtype_idx = cmd.index("--dtype")
        assert cmd[dtype_idx + 1] == expected


@pytest.mark.asyncio
async def test_build_docker_cmd_unknown_phase_raises():
    """Unknown phase raises ValueError."""
    manager = _setup_manager()
    state = CompileJobState(
        job_id="test",
        source_model="vllm/Org/Model",
        engine_name="model-test",
        gpu_indices=[0],
        dtype="fp16",
        config={},
    )

    with patch("ocabra.core.trtllm_compile_manager.settings") as s:
        s.tensorrt_llm_docker_bin = "docker"
        s.tensorrt_llm_docker_image = "trtllm:latest"
        s.tensorrt_llm_docker_models_mount_host = "/models"
        s.tensorrt_llm_docker_models_mount_container = "/data/models"

        with pytest.raises(ValueError, match="Unknown build phase"):
            manager._build_docker_cmd(state, "unknown")


# ── Multiple jobs ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_jobs_processed_sequentially():
    """Two enqueued jobs are processed one at a time."""
    manager = _setup_manager()
    completed_order = []

    convert_proc = _make_proc(returncode=0, stdout_lines=[b"ok\n"])
    build_proc = _make_proc(returncode=0, stdout_lines=[b"ok\n"])

    async def _fake_subprocess(*args, **kwargs):
        return _make_proc(returncode=0, stdout_lines=[b"ok\n"])

    with patch(
        "ocabra.core.trtllm_compile_manager.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess,
    ):
        state1 = await manager.enqueue(_make_req(engine_name="job1"))
        state2 = await manager.enqueue(_make_req(engine_name="job2"))

        task = asyncio.create_task(manager._worker_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert state1.status == "done"
    assert state2.status == "done"


# ── Engine registration ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_successful_compile_registers_engine():
    """After successful compile, _register_engine is called to add the model."""
    manager = _setup_manager()
    mock_model_manager = AsyncMock()
    manager._model_manager = mock_model_manager

    async def _fake_subprocess(*args, **kwargs):
        return _make_proc(returncode=0, stdout_lines=[b"ok\n"])

    with patch(
        "ocabra.core.trtllm_compile_manager.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess,
    ), patch.object(manager, "_cleanup_checkpoint", new=AsyncMock()):
        state = await manager.enqueue(_make_req(engine_name="test-engine"))

        task = asyncio.create_task(manager._worker_loop())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert state.status == "done"
    mock_model_manager.add_model.assert_called_once()
    call_kwargs = mock_model_manager.add_model.call_args
    assert "tensorrt_llm/test-engine" in str(call_kwargs)
