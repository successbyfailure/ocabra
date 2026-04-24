"""Background processor for OpenAI-compatible Batches API.

Polls the ``openai_batches`` table and processes ``validating`` and
``in_progress`` batches. Each request inside a batch is dispatched as an
in-process call (via ASGI transport) to the same FastAPI app, impersonating
the batch owner via ``X-Gateway-Token`` + ``X-Internal-User-Id`` headers.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
from fastapi import FastAPI
from sqlalchemy import select

from ocabra.config import settings
from ocabra.database import AsyncSessionLocal
from ocabra.db.openai_batches import OpenAIBatch, OpenAIFile

logger = structlog.get_logger(__name__)


class BatchProcessor:
    """Drains ``validating`` and ``in_progress`` batches in the background."""

    def __init__(self, app: FastAPI):
        self._app = app
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        # Internal service token used to call the FastAPI app on behalf of a
        # batch owner. We piggyback on the gateway token machinery; if no
        # gateway token is configured we generate an ephemeral one and inject
        # it into settings for this process only.
        if not settings.gateway_service_token:
            settings.gateway_service_token = secrets.token_urlsafe(32)
            logger.info("batch_processor_generated_service_token")
        self._token = settings.gateway_service_token

    def notify(self) -> None:
        """Wake the processor immediately (e.g. on batch creation)."""
        self._wake.set()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="batch-processor-loop")
        logger.info("batch_processor_started")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        self._wake.set()
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        logger.info("batch_processor_stopped")

    async def _run(self) -> None:
        interval = max(2, int(settings.batch_poll_interval_seconds))
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception as exc:
                logger.exception("batch_processor_tick_error", error=str(exc))

            try:
                await asyncio.wait_for(self._wake.wait(), timeout=interval)
            except TimeoutError:
                pass
            self._wake.clear()

    async def _tick(self) -> None:
        async with AsyncSessionLocal() as session:
            pending = (await session.execute(
                select(OpenAIBatch).where(
                    OpenAIBatch.status.in_(["validating", "in_progress", "cancelling"])
                ).order_by(OpenAIBatch.created_at)
            )).scalars().all()

        for batch in pending:
            if self._stop.is_set():
                return
            try:
                if batch.status == "validating":
                    await self._mark_in_progress(batch.id)
                    await self._process_batch(batch.id)
                elif batch.status == "in_progress":
                    await self._process_batch(batch.id)
                elif batch.status == "cancelling":
                    await self._finalize_cancellation(batch.id)
            except Exception as exc:
                logger.exception("batch_processing_error", batch_id=str(batch.id), error=str(exc))
                await self._mark_failed(batch.id, str(exc))

    async def _mark_in_progress(self, batch_id: uuid.UUID) -> None:
        async with AsyncSessionLocal() as session:
            batch = await session.get(OpenAIBatch, batch_id)
            if batch is None or batch.status != "validating":
                return
            batch.status = "in_progress"
            batch.in_progress_at = datetime.now(UTC)
            await session.commit()

    async def _mark_failed(self, batch_id: uuid.UUID, message: str) -> None:
        async with AsyncSessionLocal() as session:
            batch = await session.get(OpenAIBatch, batch_id)
            if batch is None:
                return
            batch.status = "failed"
            batch.failed_at = datetime.now(UTC)
            batch.errors = {"data": [{"code": "processing_error", "message": message}]}
            await session.commit()

    async def _finalize_cancellation(self, batch_id: uuid.UUID) -> None:
        async with AsyncSessionLocal() as session:
            batch = await session.get(OpenAIBatch, batch_id)
            if batch is None:
                return
            batch.status = "cancelled"
            batch.cancelled_at = datetime.now(UTC)
            await session.commit()

    async def _process_batch(self, batch_id: uuid.UUID) -> None:
        # Load batch + input file path + user
        async with AsyncSessionLocal() as session:
            batch = await session.get(OpenAIBatch, batch_id)
            if batch is None:
                return
            if batch.status not in ("in_progress",):
                return
            input_file = await session.get(OpenAIFile, batch.input_file_id)
            if input_file is None:
                raise RuntimeError("Input file metadata missing")
            input_path = Path(input_file.storage_path)
            user_id = str(batch.user_id)
            api_key_name = ""
            if batch.api_key_id is not None:
                from ocabra.db.auth import ApiKey

                row = (await session.execute(
                    select(ApiKey.name).where(ApiKey.id == batch.api_key_id)
                )).first()
                if row:
                    api_key_name = row[0]
            endpoint = batch.endpoint

        if not input_path.exists():
            raise RuntimeError("Input file content missing on disk")

        # Prepare output file on disk
        output_uuid = uuid.uuid4()
        storage_dir = Path(settings.openai_files_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        output_path = storage_dir / f"{output_uuid}"

        concurrency = max(1, int(settings.batch_max_concurrency))
        sem = asyncio.Semaphore(concurrency)

        # Read all lines first (batches are bounded; large batches assumed OK
        # for current scale). For very large files this could stream instead.
        with input_path.open("r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]

        # Skip already-processed lines on resume by counting (best-effort: we
        # don't persist per-line progress, so on resume we re-run unfinished
        # work; simplification accepted given the workload size).
        results_lock = asyncio.Lock()
        completed = 0
        failed = 0

        # Open output for append (so resume semantics still produce one file)
        out_handle = output_path.open("w", encoding="utf-8")

        cancelled = False

        async def _process_line(idx: int, line: str) -> None:
            nonlocal completed, failed, cancelled
            if cancelled:
                return
            async with sem:
                if cancelled:
                    return
                # Check cancellation before each line
                if await self._is_cancelling(batch_id):
                    cancelled = True
                    return
                try:
                    obj = json.loads(line)
                except Exception as exc:
                    record = self._error_record(
                        custom_id=None,
                        status_code=400,
                        error_type="invalid_request_error",
                        message=f"Invalid JSON: {exc}",
                    )
                    async with results_lock:
                        out_handle.write(json.dumps(record) + "\n")
                        failed += 1
                    return

                custom_id = obj.get("custom_id") or f"req_{idx}"
                body = obj.get("body") or {}

                response_payload, status_code, error_payload = await self._dispatch(
                    endpoint=endpoint,
                    body=body,
                    user_id=user_id,
                    api_key_name=api_key_name,
                )

                record: dict[str, Any] = {
                    "id": f"batch_req_{uuid.uuid4()}",
                    "custom_id": custom_id,
                }
                if error_payload is not None:
                    record["response"] = None
                    record["error"] = error_payload
                    async with results_lock:
                        out_handle.write(json.dumps(record) + "\n")
                        failed += 1
                else:
                    record["response"] = {
                        "status_code": status_code,
                        "request_id": str(uuid.uuid4()),
                        "body": response_payload,
                    }
                    record["error"] = None
                    async with results_lock:
                        out_handle.write(json.dumps(record) + "\n")
                        completed += 1

        # Periodic checkpoint so the user can see progress
        async def _checkpoint() -> None:
            while not cancelled:
                await asyncio.sleep(5)
                async with results_lock:
                    c, f = completed, failed
                async with AsyncSessionLocal() as s:
                    b = await s.get(OpenAIBatch, batch_id)
                    if b is None or b.status != "in_progress":
                        return
                    b.request_completed = c
                    b.request_failed = f
                    await s.commit()

        checkpoint_task = asyncio.create_task(_checkpoint(), name=f"batch-{batch_id}-checkpoint")

        try:
            await asyncio.gather(*(
                _process_line(idx, line) for idx, line in enumerate(lines)
            ))
        finally:
            checkpoint_task.cancel()
            with suppress(asyncio.CancelledError):
                await checkpoint_task
            out_handle.close()

        # Finalize
        async with AsyncSessionLocal() as session:
            batch = await session.get(OpenAIBatch, batch_id)
            if batch is None:
                return

            if cancelled or batch.status == "cancelling":
                batch.status = "cancelled"
                batch.cancelled_at = datetime.now(UTC)
                # Clean up partial output
                output_path.unlink(missing_ok=True)
                batch.request_completed = completed
                batch.request_failed = failed
                await session.commit()
                return

            batch.finalizing_at = datetime.now(UTC)
            batch.request_completed = completed
            batch.request_failed = failed

            output_size = output_path.stat().st_size if output_path.exists() else 0
            output_record = OpenAIFile(
                id=output_uuid,
                user_id=batch.user_id,
                filename=f"batch_{batch.id}_output.jsonl",
                bytes=output_size,
                purpose="batch_output",
                storage_path=str(output_path),
                status="processed",
            )
            session.add(output_record)
            await session.flush()  # ensure openai_files row exists before FK reference
            batch.output_file_id = output_uuid
            batch.status = "completed"
            batch.completed_at = datetime.now(UTC)
            await session.commit()
            logger.info(
                "batch_completed",
                batch_id=str(batch_id),
                completed=completed,
                failed=failed,
            )

    async def _is_cancelling(self, batch_id: uuid.UUID) -> bool:
        async with AsyncSessionLocal() as session:
            batch = await session.get(OpenAIBatch, batch_id)
            return batch is not None and batch.status == "cancelling"

    async def _dispatch(
        self,
        endpoint: str,
        body: dict[str, Any],
        user_id: str,
        api_key_name: str,
    ) -> tuple[dict[str, Any] | None, int, dict[str, Any] | None]:
        """Call the FastAPI app in-process and return (body, status, error)."""
        # Force non-streaming for batch
        body = dict(body)
        body.pop("stream", None)
        body.pop("stream_options", None)

        headers = {
            "X-Gateway-Token": self._token,
            "X-Internal-User-Id": user_id,
            "Content-Type": "application/json",
        }
        if api_key_name:
            headers["X-Internal-Key-Name"] = api_key_name

        transport = httpx.ASGITransport(app=self._app)
        try:
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://batch-processor.local",
                timeout=settings.batch_request_timeout_seconds,
            ) as client:
                response = await client.post(endpoint, json=body, headers=headers)
        except Exception as exc:
            return None, 500, {
                "code": "internal_error",
                "message": f"Internal dispatch failed: {exc}",
            }

        if response.status_code >= 400:
            try:
                err = response.json()
            except Exception:
                err = {"message": response.text}
            error_payload = err.get("error") if isinstance(err, dict) and "error" in err else err
            if not isinstance(error_payload, dict):
                error_payload = {"message": str(error_payload)}
            error_payload.setdefault("code", f"http_{response.status_code}")
            return None, response.status_code, error_payload

        try:
            payload = response.json()
        except Exception as exc:
            return None, response.status_code, {
                "code": "invalid_response",
                "message": f"Worker returned non-JSON response: {exc}",
            }
        return payload, response.status_code, None

    @staticmethod
    def _error_record(
        custom_id: str | None,
        status_code: int,
        error_type: str,
        message: str,
    ) -> dict[str, Any]:
        return {
            "id": f"batch_req_{uuid.uuid4()}",
            "custom_id": custom_id,
            "response": None,
            "error": {
                "code": error_type,
                "message": message,
            },
        }
