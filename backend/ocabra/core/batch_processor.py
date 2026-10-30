"""Background processor for OpenAI-compatible Batches API.

Polls the ``openai_batches`` table and processes ``validating`` and
``in_progress`` batches. Each request inside a batch is dispatched as an
in-process call (via ASGI transport) to the same FastAPI app, impersonating
the batch owner via ``X-Gateway-Token`` + ``X-Internal-User-Id`` headers,
tagged ``X-Internal-Request-Origin: batch`` so the fair-share gate and the
batch-vs-interactive priority coordinator (``batch_priority.py``) can tell
this traffic apart from a real client hitting the same endpoint.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import secrets
import time
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

    def __init__(self, app: FastAPI, coordinator: Any | None = None):
        self._app = app
        self._coordinator = coordinator
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

        cooldown_s = max(0.0, float(settings.batch_inter_job_cooldown_s))
        for i, batch in enumerate(pending):
            if self._stop.is_set():
                return
            # Expired while this processor wasn't looking at it (down for a
            # while, or just never got a turn) — finalize as expired without
            # dispatching anything new. A batch that expires WHILE actively
            # being processed is caught inside _process_batch's own loop
            # instead (same expires_at, checked against wall clock there).
            if (
                batch.status in ("validating", "in_progress")
                and batch.expires_at is not None
                and datetime.now(UTC) > batch.expires_at
            ):
                await self._finalize_expiration(batch.id)
                continue
            ran_a_job = False
            try:
                if batch.status == "validating":
                    await self._mark_in_progress(batch.id)
                    await self._process_batch(batch.id)
                    ran_a_job = True
                elif batch.status == "in_progress":
                    await self._process_batch(batch.id)
                    ran_a_job = True
                elif batch.status == "cancelling":
                    await self._finalize_cancellation(batch.id)
            except Exception as exc:
                logger.exception("batch_processing_error", batch_id=str(batch.id), error=str(exc))
                await self._mark_failed(batch.id, str(exc))

            # Let the server breathe between back-to-back long-running batch
            # jobs — unrelated to interactive priority, just avoids pegging
            # the GPUs continuously across consecutive big batches. Only
            # worth sleeping if there's another job queued right behind this
            # one in the same tick; otherwise the next poll interval already
            # provides the gap.
            if ran_a_job and cooldown_s > 0 and i + 1 < len(pending) and not self._stop.is_set():
                logger.info("batch_inter_job_cooldown", seconds=cooldown_s)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=cooldown_s)
                except TimeoutError:
                    pass

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

    async def _finalize_expiration(self, batch_id: uuid.UUID) -> None:
        """Mark a batch ``expired`` when its ``completion_window`` has
        elapsed. Keeps whatever output/error files already exist (partial
        results from lines that did finish) rather than discarding them —
        matches the real OpenAI contract: an expired batch still exposes
        ``output_file_id``/``error_file_id`` for whatever completed before
        the deadline. A batch that never got past ``validating`` has no
        files yet; nothing to keep in that case.
        """
        async with AsyncSessionLocal() as session:
            batch = await session.get(OpenAIBatch, batch_id)
            if batch is None or batch.status not in ("validating", "in_progress"):
                return
            batch.status = "expired"
            batch.expired_at = datetime.now(UTC)
            await session.commit()
            logger.info(
                "batch_expired",
                batch_id=str(batch_id),
                had_output=batch.output_file_id is not None,
            )

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
            expires_at = batch.expires_at

            # 2026-10-30 — crash-survival watermark. Persist the output file's
            # identity on the batch row BEFORE any dispatch happens, instead
            # of only at completion. A process restart mid-batch (container
            # redeploy, OOM, etc.) then resumes against the SAME output file
            # path and skips whatever's already in it, rather than starting
            # a brand-new output file and silently orphaning the previous
            # partial one while redispatching everything from line 0 (the
            # original, documented limitation of this processor).
            is_resume = batch.output_file_id is not None
            output_uuid = batch.output_file_id
            error_uuid = batch.error_file_id
            needs_commit = False
            if output_uuid is None:
                output_uuid = uuid.uuid4()
                session.add(OpenAIFile(
                    id=output_uuid,
                    user_id=batch.user_id,
                    filename=f"batch_{batch.id}_output.jsonl",
                    bytes=0,
                    purpose="batch_output",
                    storage_path=str(Path(settings.openai_files_dir) / f"{output_uuid}"),
                    status="processing",
                ))
                needs_commit = True
            if error_uuid is None:
                # Also covers a batch resumed from before this output/error
                # split existed (``batch.output_file_id`` already set,
                # ``error_file_id`` still None) — allocated here the same
                # way regardless of whether this is a fresh batch or a
                # resume missing just this one.
                error_uuid = uuid.uuid4()
                session.add(OpenAIFile(
                    id=error_uuid,
                    user_id=batch.user_id,
                    filename=f"batch_{batch.id}_errors.jsonl",
                    bytes=0,
                    purpose="batch_error",
                    storage_path=str(Path(settings.openai_files_dir) / f"{error_uuid}"),
                    status="processing",
                ))
                needs_commit = True
            if needs_commit:
                # Flush the INSERTs before the UPDATE that references them —
                # SQLAlchemy doesn't know these are related (plain scalar FK
                # columns, no relationship()), so without this the unit of
                # work can flush the batch UPDATE first and the FK rejects
                # it (reproduced live 2026-10-30). Same guard the original
                # finalize-time code already had; just needed here too now
                # that this runs before dispatch, not only after.
                await session.flush()
                batch.output_file_id = output_uuid
                batch.error_file_id = error_uuid
                await session.commit()

        if not input_path.exists():
            raise RuntimeError("Input file content missing on disk")

        storage_dir = Path(settings.openai_files_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        output_path = storage_dir / f"{output_uuid}"
        error_path = storage_dir / f"{error_uuid}"

        concurrency = max(1, int(settings.batch_max_concurrency))
        sem = asyncio.Semaphore(concurrency)

        # Read all lines first (batches are bounded; large batches assumed OK
        # for current scale). For very large files this could stream instead.
        with input_path.open("r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]

        def _custom_id_for(idx: int, line: str) -> str:
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                return f"req_{idx}"
            return (isinstance(obj, dict) and obj.get("custom_id")) or f"req_{idx}"

        results_lock = asyncio.Lock()
        completed = 0
        failed = 0
        done_custom_ids: set[str] = set()
        if is_resume:
            for path, is_error_file in ((output_path, False), (error_path, True)):
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as f:
                    for out_line in f:
                        out_line = out_line.strip()
                        if not out_line:
                            continue
                        try:
                            rec = json.loads(out_line)
                        except Exception:  # noqa: BLE001
                            continue
                        cid = rec.get("custom_id")
                        if cid:
                            done_custom_ids.add(cid)
                        if is_error_file:
                            failed += 1
                        else:
                            completed += 1
            logger.info(
                "batch_resumed_from_watermark",
                batch_id=str(batch_id),
                already_done=len(done_custom_ids),
                total_lines=len(lines),
            )

        # Open for append — on a fresh batch the file doesn't exist yet (this
        # creates it); on resume it continues the same file instead of
        # truncating what crash-survival just recovered. Success records go
        # to ``out_handle``/output_file_id, failures to ``err_handle``/
        # error_file_id — matching the real OpenAI Batches contract (2026-
        # 10-30: previously both were mixed into the same output file).
        out_handle = output_path.open("a", encoding="utf-8")
        err_handle = error_path.open("a", encoding="utf-8")

        cancelled = False
        coordinator = self._coordinator
        profile_registry = getattr(self._app.state, "profile_registry", None)
        router_resolver = getattr(self._app.state, "router_resolver", None)
        # profile_id -> base_model_id resolution for the priority
        # coordinator's registry key only (NOT used for the real dispatch,
        # which still sends the user's original ``model`` field untouched —
        # this never changes what actually gets served, only what the
        # coordinator thinks the line is targeting).
        #
        # A PLAIN profile resolves once to a stable base_model_id and is
        # cached. A ROUTER profile's target can change between lines (warm
        # state shifts, an eviction happens mid-batch) so it's intentionally
        # re-resolved every call via the SAME RouterResolver.pick() the real
        # dispatch uses — duplicating that decision work, but a stale cached
        # router target would be a worse bookkeeping key than a fresh one.
        # 2026-10-30: this used to not resolve routers at all (documented
        # limitation); fixed since router-backed batch lines are exactly
        # the kind of traffic worth prioritizing correctly against.
        _model_key_cache: dict[str, str] = {}

        async def _coordinator_key(raw_model: str, body: dict) -> str:
            if raw_model in _model_key_cache:
                return _model_key_cache[raw_model]
            if profile_registry is None:
                return raw_model
            try:
                profile = await profile_registry.get(raw_model)
            except Exception:  # noqa: BLE001 — coordinator key only, never block dispatch
                return raw_model
            if profile is None:
                return raw_model
            if router_resolver is not None:
                try:
                    if router_resolver.is_router(profile):
                        resolved_target = await router_resolver.pick(profile, request_body=body)
                        return resolved_target.base_model_id  # not cached — can shift between lines
                except Exception:  # noqa: BLE001
                    pass
            resolved = profile.base_model_id
            _model_key_cache[raw_model] = resolved
            return resolved

        async def _process_line(idx: int, line: str) -> bool:
            """Returns True once the line is fully resolved (success or a
            permanent error written to the output file). Returns False when
            it needs another attempt — skipped because its target model is
            in a post-preemption cooldown, or killed mid-flight by
            interactive priority (see ``batch_priority.py``) — the caller
            requeues it instead of treating it as a failure.
            """
            nonlocal completed, failed, cancelled
            if cancelled:
                return True
            async with sem:
                if cancelled:
                    return True
                # Check cancellation before each line
                if await self._is_cancelling(batch_id):
                    cancelled = True
                    return True
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
                        err_handle.write(json.dumps(record) + "\n")
                        failed += 1
                    return True

                custom_id = obj.get("custom_id") or f"req_{idx}"
                body = obj.get("body") or {}
                # Key for the priority coordinator — resolved from whatever
                # the user wrote in ``model`` (bare model_id, profile_id, or
                # router profile_id) to the same base_model_id
                # chat.py/completions.py land on. BatchPriorityCoordinator
                # additionally strips any ``::<hash>`` runtime-preset suffix
                # on both sides (the real worker_key is e.g.
                # ``llama_cpp/qwen3.8-27b::781a0367a8ba``).
                target_model = await _coordinator_key(str(body.get("model") or ""), body)

                if coordinator is not None and target_model and coordinator.in_cooldown(target_model):
                    return False

                handle = None
                if coordinator is not None and target_model:
                    handle = await coordinator.register(
                        target_model, asyncio.current_task(), batch_id=str(batch_id), line_idx=idx
                    )
                    logger.info(
                        "batch_priority_registered",
                        target_model=target_model,
                        batch_id=str(batch_id),
                        line_idx=idx,
                        handle=handle,
                    )
                try:
                    response_payload, status_code, error_payload = await self._dispatch(
                        endpoint=endpoint,
                        body=body,
                        user_id=user_id,
                        api_key_name=api_key_name,
                    )
                except asyncio.CancelledError:
                    # Preempted to free a slot/model for a waiting
                    # interactive caller — not a real failure. Swallow it
                    # here (the task itself tolerates being cancelled; see
                    # BatchPriorityCoordinator.preempt_now) and let the
                    # caller retry the line once the cooldown lifts.
                    return False
                finally:
                    if handle is not None:
                        await coordinator.unregister(target_model, handle)

                record: dict[str, Any] = {
                    "id": f"batch_req_{uuid.uuid4()}",
                    "custom_id": custom_id,
                }
                if error_payload is not None:
                    record["response"] = None
                    record["error"] = error_payload
                    async with results_lock:
                        err_handle.write(json.dumps(record) + "\n")
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
                return True

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

        # Lines skipped (cooldown) or killed (preemption) come back here for
        # another attempt rather than being written as permanent failures.
        # Bounded by wall clock, not a pass count: a fixed pass cap sized for
        # the slot cooldown (120s default) can run out almost exactly as a
        # longer eviction cooldown (300s default) is still active — caught
        # live 2026-10-30 (a real eviction fired mid-test and very nearly hit
        # this). Give comfortable headroom above whichever cooldown is
        # configured longest, since both are user-configurable settings.
        _REQUEUE_BACKOFF_S = 5.0
        requeue_deadline = time.monotonic() + max(
            settings.batch_resume_cooldown_slots_s,
            settings.batch_resume_cooldown_eviction_s,
        ) + 60.0

        expired_mid_run = False
        try:
            pending_indices = [
                idx for idx, line in enumerate(lines)
                if _custom_id_for(idx, line) not in done_custom_ids
            ]
            while pending_indices and not cancelled and time.monotonic() < requeue_deadline:
                if expires_at is not None and datetime.now(UTC) > expires_at:
                    expired_mid_run = True
                    break
                results = await asyncio.gather(*(
                    _process_line(idx, lines[idx]) for idx in pending_indices
                ))
                pending_indices = [
                    idx for idx, done in zip(pending_indices, results) if not done
                ]
                if pending_indices and not cancelled:
                    await asyncio.sleep(_REQUEUE_BACKOFF_S)

            if pending_indices and not cancelled:
                if expired_mid_run:
                    logger.warning(
                        "batch_expired_mid_run",
                        batch_id=str(batch_id),
                        remaining=len(pending_indices),
                    )
                    error_type = "batch_expired"
                    message = (
                        "The batch's completion_window elapsed before this request "
                        "was dispatched."
                    )
                else:
                    logger.warning(
                        "batch_requeue_exhausted",
                        batch_id=str(batch_id),
                        remaining=len(pending_indices),
                    )
                    error_type = "batch_priority_preempted"
                    message = (
                        "Repeatedly preempted to free capacity for interactive "
                        "traffic and gave up retrying within this batch run."
                    )
                for idx in pending_indices:
                    record = self._error_record(
                        custom_id=_custom_id_for(idx, lines[idx]),
                        status_code=503,
                        error_type=error_type,
                        message=message,
                    )
                    async with results_lock:
                        err_handle.write(json.dumps(record) + "\n")
                        failed += 1
        finally:
            checkpoint_task.cancel()
            with suppress(asyncio.CancelledError):
                await checkpoint_task
            out_handle.close()
            err_handle.close()

        # Finalize
        async with AsyncSessionLocal() as session:
            batch = await session.get(OpenAIBatch, batch_id)
            if batch is None:
                return

            if cancelled or batch.status == "cancelling":
                batch.status = "cancelled"
                batch.cancelled_at = datetime.now(UTC)
                # Clean up the partial output/error files — both on disk and
                # the placeholder OpenAIFile rows created early for the
                # crash-survival watermark (ON DELETE SET NULL on the FK).
                output_path.unlink(missing_ok=True)
                error_path.unlink(missing_ok=True)
                for fid in (output_uuid, error_uuid):
                    record = await session.get(OpenAIFile, fid)
                    if record is not None:
                        await session.delete(record)
                batch.output_file_id = None
                batch.error_file_id = None
                batch.request_completed = completed
                batch.request_failed = failed
                await session.commit()
                return

            batch.finalizing_at = datetime.now(UTC)
            batch.request_completed = completed
            batch.request_failed = failed

            # The OpenAIFile rows for output/error already exist (created
            # before dispatch for the crash-survival watermark) — update in
            # place rather than inserting a second row under the same id.
            output_size = output_path.stat().st_size if output_path.exists() else 0
            output_record = await session.get(OpenAIFile, output_uuid)
            if output_record is not None:
                output_record.bytes = output_size
                output_record.status = "processed"
            if failed > 0:
                error_size = error_path.stat().st_size if error_path.exists() else 0
                error_record = await session.get(OpenAIFile, error_uuid)
                if error_record is not None:
                    error_record.bytes = error_size
                    error_record.status = "processed"
            else:
                # No failures — match the real OpenAI contract (error_file_id
                # stays null when nothing failed) instead of pointing at an
                # empty placeholder file.
                error_path.unlink(missing_ok=True)
                error_record = await session.get(OpenAIFile, error_uuid)
                if error_record is not None:
                    await session.delete(error_record)
                batch.error_file_id = None
            if expired_mid_run:
                batch.status = "expired"
                batch.expired_at = datetime.now(UTC)
            else:
                batch.status = "completed"
                batch.completed_at = datetime.now(UTC)
            await session.commit()
            logger.info(
                "batch_completed" if not expired_mid_run else "batch_expired",
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
            "X-Internal-Request-Origin": "batch",
        }
        if api_key_name:
            headers["X-Internal-Key-Name"] = api_key_name

        transport = httpx.ASGITransport(app=self._app)
        dispatch_timeout_s = max(
            int(settings.batch_request_timeout_seconds),
            int(settings.inference_request_timeout_seconds) + 30,
        )

        multipart_error = None
        files_kw: dict[str, Any] | None = None
        data_kw: dict[str, Any] | None = None
        json_kw: dict[str, Any] | None = body
        if endpoint == "/v1/audio/transcriptions":
            # oCabra extension (2026-10-30): the real endpoint takes
            # multipart/form-data with a binary file, not JSON — the batch
            # line carries a file_id (pre-uploaded via /v1/files,
            # purpose="batch_input_audio") instead of inline audio. Rebuild
            # the real multipart request from that reference here.
            json_kw = None
            raw_file_id = str(body.pop("file_id", ""))
            audio_id = raw_file_id[5:] if raw_file_id.startswith("file-") else raw_file_id
            try:
                async with AsyncSessionLocal() as session:
                    audio_file = await session.get(OpenAIFile, uuid.UUID(audio_id))
            except ValueError:
                audio_file = None
            if audio_file is None:
                multipart_error = {
                    "code": "invalid_file_id",
                    "message": f"body.file_id '{raw_file_id}' does not reference an uploaded file",
                }
            else:
                audio_path = Path(audio_file.storage_path)
                if not audio_path.exists():
                    multipart_error = {
                        "code": "file_not_found",
                        "message": "Referenced audio file content missing on disk",
                    }
                else:
                    content_type = mimetypes.guess_type(audio_file.filename)[0] or "application/octet-stream"
                    files_kw = {"file": (audio_file.filename, audio_path.read_bytes(), content_type)}
                    # dict value = list means "repeat this field" to httpx's
                    # MultipartStream (same idiom as encode_urlencoded_data) —
                    # matches what transcriptions() reads via
                    # form.getlist("timestamp_granularities[]"). A plain
                    # list/tuple passed as the whole `data=` argument instead
                    # of a dict silently takes httpx's deprecated
                    # "data=<raw bytes>" compat path and produces a SYNC
                    # stream — reproduced live 2026-10-30 ("Attempted to send
                    # a sync request with an AsyncClient instance").
                    data_kw = {}
                    for key, value in body.items():
                        if value is None:
                            continue
                        if key == "timestamp_granularities" and isinstance(value, list):
                            data_kw["timestamp_granularities[]"] = [str(v) for v in value]
                        else:
                            data_kw[key] = str(value)

        if multipart_error is not None:
            return None, 400, multipart_error

        try:
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://batch-processor.local",
                timeout=dispatch_timeout_s,
            ) as client:
                if files_kw is not None:
                    response = await client.post(endpoint, files=files_kw, data=data_kw, headers=headers)
                else:
                    response = await client.post(
                        endpoint, json=json_kw, headers={**headers, "Content-Type": "application/json"}
                    )
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

        content_type = response.headers.get("content-type", "")
        if "json" in content_type.split(";")[0]:
            try:
                payload = response.json()
            except Exception as exc:
                return None, response.status_code, {
                    "code": "invalid_response",
                    "message": f"Worker returned non-JSON response: {exc}",
                }
            return payload, response.status_code, None

        # Binary artifact (e.g. /v1/audio/speech returns raw audio bytes,
        # not JSON) — can't inline it in the batch's output JSONL the way a
        # chat/embeddings/images response fits. Save it as its own file and
        # reference it by file_id instead, reusing the same Files API the
        # rest of the Batches output already relies on; the client downloads
        # it via GET /v1/files/{id}/content like any other file.
        raw_bytes = response.content
        artifact_uuid = uuid.uuid4()
        storage_dir = Path(settings.openai_files_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = storage_dir / f"{artifact_uuid}"
        try:
            artifact_path.write_bytes(raw_bytes)
            async with AsyncSessionLocal() as session:
                session.add(OpenAIFile(
                    id=artifact_uuid,
                    user_id=uuid.UUID(user_id),
                    filename=f"batch_artifact_{artifact_uuid}",
                    bytes=len(raw_bytes),
                    purpose="batch_artifact",
                    storage_path=str(artifact_path),
                    status="processed",
                ))
                await session.commit()
        except Exception as exc:
            return None, 500, {
                "code": "artifact_write_failed",
                "message": f"Failed to persist binary response: {exc}",
            }
        payload = {
            "file_id": f"file-{artifact_uuid}",
            "bytes": len(raw_bytes),
            "content_type": content_type.split(";")[0] or "application/octet-stream",
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
