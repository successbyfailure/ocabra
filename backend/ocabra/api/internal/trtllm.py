"""TensorRT-LLM compilation API.

Endpoints:
    POST   /ocabra/trtllm/compile              — submit a compile job
    GET    /ocabra/trtllm/compile               — list all jobs (history)
    GET    /ocabra/trtllm/compile/{job_id}/stream — SSE progress stream
    DELETE /ocabra/trtllm/compile/{job_id}      — cancel a job
"""
from __future__ import annotations

import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ocabra.core.trtllm_compile_manager import CompileRequest
from ocabra.redis_client import subscribe

router = APIRouter(tags=["trtllm"])

# ── Pydantic schemas ─────────────────────────────────────────────


class CompileJobCreate(BaseModel):
    """Request body for POST /ocabra/trtllm/compile."""

    model_id: str = Field(
        ...,
        description="Source model identifier, e.g. 'vllm/Qwen/Qwen3.5-27B-GPTQ-Int4'",
    )
    gpu_indices: list[int] = Field(
        ...,
        description="GPU indices to use, e.g. [1] or [0, 1] for tensor parallelism",
    )
    dtype: str = Field(
        "fp16",
        description="Weight dtype: fp16 | bf16 | int8 | fp8",
    )
    max_batch_size: Annotated[int, Field(ge=1, le=256)] = 1
    max_input_len: Annotated[int, Field(ge=128)] = 2048
    max_seq_len: Annotated[int, Field(ge=256)] = 4096
    engine_name: str = Field(
        ...,
        description="Name for the compiled engine (used as directory name)",
    )


# ── Endpoints ────────────────────────────────────────────────────


@router.post("/trtllm/compile", status_code=202)
async def create_compile_job(body: CompileJobCreate, request: Request) -> dict:
    """Submit a TensorRT-LLM engine compilation job.

    Parameters:
        body: CompileJobCreate — compilation parameters.

    Returns:
        dict with job_id and stream_url for SSE progress.
    """
    manager = _get_manager(request)

    req = CompileRequest(
        source_model=body.model_id,
        engine_name=body.engine_name,
        gpu_indices=body.gpu_indices,
        dtype=body.dtype,
        max_batch_size=body.max_batch_size,
        max_input_len=body.max_input_len,
        max_seq_len=body.max_seq_len,
    )
    state = await manager.enqueue(req)
    return {
        "job_id": state.job_id,
        "stream_url": f"/ocabra/trtllm/compile/{state.job_id}/stream",
        **state.to_dict(),
    }


@router.get("/trtllm/compile")
async def list_compile_jobs(request: Request) -> list[dict]:
    """List all TensorRT-LLM compilation jobs (history + active).

    Returns:
        List of job dicts ordered by started_at descending.
    """
    manager = _get_manager(request)
    jobs = await manager.list_jobs()
    return [j.to_dict() for j in jobs]


@router.get("/trtllm/compile/{job_id}/stream")
async def stream_compile_job(job_id: str, request: Request) -> StreamingResponse:
    """Stream real-time progress for a compilation job via SSE.

    Subscribes to the Redis pub/sub channel for the job and forwards
    events until the job reaches a terminal state or the client disconnects.

    Parameters:
        job_id: Compile job identifier.

    Returns:
        Server-Sent Events stream with progress and log events.
    """
    manager = _get_manager(request)

    # Validate job exists
    state = manager.get_job(job_id)
    if state is None:
        jobs = await manager.list_jobs()
        match = next((j for j in jobs if j.job_id == job_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        state = match

    # If already terminal, return final state immediately
    if state.status in {"done", "failed", "cancelled"}:
        async def _final_stream():
            data = json.dumps(state.to_dict())
            yield f"data: {data}\n\n"

        return StreamingResponse(_final_stream(), media_type="text/event-stream")

    async def _event_generator():
        channel = f"trtllm:compile:{job_id}"
        async with subscribe(channel) as pubsub:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(pubsub.get_message(ignore_subscribe_messages=True), timeout=1.0)
                except TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
                    continue

                if msg is None:
                    yield ": keepalive\n\n"
                    continue

                data = msg.get("data", "")
                if not data:
                    continue

                yield f"data: {data}\n\n"

                # Stop streaming when job reaches terminal state
                try:
                    parsed = json.loads(data)
                    if parsed.get("status") in {"done", "failed", "cancelled"}:
                        break
                except (json.JSONDecodeError, AttributeError):
                    pass

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/trtllm/compile/{job_id}", status_code=200)
async def cancel_compile_job(job_id: str, request: Request) -> dict:
    """Cancel a pending or running compilation job.

    Parameters:
        job_id: Compile job identifier.

    Returns:
        Updated job dict with status='cancelled'.
    """
    manager = _get_manager(request)
    try:
        state = await manager.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return state.to_dict()


# ── Helpers ──────────────────────────────────────────────────────


def _get_manager(request: Request):
    manager = getattr(request.app.state, "trtllm_compile_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="TrtllmCompileManager not initialised",
        )
    return manager
