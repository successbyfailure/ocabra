"""
POST /api/pull — delegate model pull to DownloadManager and stream NDJSON progress.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from ocabra.api.internal.downloads import download_manager
from ocabra.redis_client import subscribe

router = APIRouter()


class PullRequest(BaseModel):
    name: str
    stream: bool = True


@router.post("/pull", summary="Pull a model")
async def pull_model(body: PullRequest):
    """
    Queue a model download and report pull progress in Ollama NDJSON format.

    Parameters:
      - name: model reference in Ollama format.
      - stream: if true, return an NDJSON stream.
    """
    try:
        job = await download_manager.enqueue(source="ollama", model_ref=body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    if not body.stream:
        while True:
            current = await download_manager.get_job(job.job_id)
            if current is None:
                raise HTTPException(status_code=500, detail={"error": "download job disappeared"})
            if current.status in {"completed", "failed", "cancelled"}:
                return _job_to_payload(current.model_dump(mode="json"))
            await asyncio.sleep(0.2)

    return StreamingResponse(
        _stream_pull_progress(job.job_id),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_pull_progress(job_id: str) -> AsyncIterator[bytes]:
    # Emit a first line to match Ollama behavior.
    yield b'{"status":"pulling manifest"}\n'

    channel = f"download:progress:{job_id}"
    try:
        async with subscribe(channel) as pubsub:
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    await asyncio.sleep(0.05)
                    continue

                data = message.get("data")
                if not isinstance(data, str):
                    continue

                payload = json.loads(data)
                out = _job_to_payload(payload)
                yield (json.dumps(out) + "\n").encode("utf-8")

                status = payload.get("status")
                if status in {"completed", "failed", "cancelled"}:
                    break
    except RuntimeError:
        # If Redis is not initialized (e.g. tests), poll directly.
        while True:
            current = await download_manager.get_job(job_id)
            if current is None:
                break
            out = _job_to_payload(current.model_dump(mode="json"))
            yield (json.dumps(out) + "\n").encode("utf-8")
            if current.status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.2)


def _job_to_payload(job: dict) -> dict:
    status = str(job.get("status") or "")
    if status == "queued":
        return {"status": "pulling manifest"}
    if status == "downloading":
        pct = float(job.get("progress_pct") or 0.0)
        return {
            "status": "pulling layer",
            "completed": int(pct * 1000),
            "total": 100000,
        }
    if status == "completed":
        return {"status": "success"}
    if status == "failed":
        return {"status": "error", "error": job.get("error") or "download failed"}
    if status == "cancelled":
        return {"status": "cancelled"}

    return {
        "status": status or "unknown",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
