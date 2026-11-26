"""Peticiones de máquina que se reintentan hasta que el proveedor tiene capacidad.

Cuando se va a lanzar un pod, a menudo la GPU elegida no está libre (RunPod
contesta «no hay máquinas»; su catálogo de stock es aproximado). En vez de
volver a intentarlo a mano, se deja una petición: el bucle de nube la prueba
cada ``retry_interval_s`` con cada GPU aceptable, por orden de preferencia,
hasta que lanza un pod o caduca.

Solo se reintenta la falta de capacidad (:class:`CloudCapacityError`). Los
demás fallos —presupuesto agotado, sin API key, volumen en uso, imagen— cierran
la petición como ``failed`` con su motivo: reintentarlos no los arregla.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select

from ocabra.core.cloud.manager import CloudCapacityError, CloudError

logger = structlog.get_logger(__name__)

ACTIVE = ("waiting",)
TERMINAL = ("launched", "expired", "cancelled", "failed")


def _now() -> datetime:
    return datetime.now(UTC)


def request_out(row: Any) -> dict[str, Any]:
    launch = dict(row.launch or {})
    return {
        "id": str(row.id),
        "status": row.status,
        "gpuTypeIds": list(row.gpu_type_ids or []),
        "gpuCount": launch.get("gpu_count", 1),
        "cloudType": launch.get("cloud_type"),
        "name": launch.get("name"),
        "models": launch.get("models") or [],
        "recipes": launch.get("recipes") or [],
        "backends": launch.get("backends") or [],
        "retryIntervalS": row.retry_interval_s,
        "expiresAt": row.expires_at.isoformat() if row.expires_at else None,
        "attempts": row.attempts,
        "lastAttemptAt": row.last_attempt_at.isoformat() if row.last_attempt_at else None,
        "lastError": row.last_error,
        "cloudNodeId": str(row.cloud_node_id) if row.cloud_node_id else None,
        "launchedGpuTypeId": row.launched_gpu_type_id,
        "purpose": row.purpose,
        "benchmarkRunId": str(row.benchmark_run_id) if row.benchmark_run_id else None,
        "createdAt": row.created_at.isoformat() if row.created_at else None,
        "finishedAt": row.finished_at.isoformat() if row.finished_at else None,
    }


class LaunchRequestQueue:
    """Cola de peticiones de máquina de un :class:`CloudNodeManager`."""

    def __init__(self, cloud: Any) -> None:
        self._cloud = cloud

    @property
    def _sessions(self):
        return self._cloud.fm.session_factory

    async def create(
        self,
        *,
        gpu_type_ids: list[str],
        launch: dict[str, Any],
        wait_minutes: int,
        retry_interval_s: int,
        purpose: str = "manual",
        benchmark_run_id: uuid.UUID | None = None,
        try_now: bool = True,
    ) -> dict[str, Any]:
        """Crea la petición y, si ``try_now``, hace el primer intento ya."""
        from ocabra.db.cloud import CloudLaunchRequest

        gpus = [gpu for gpu in dict.fromkeys(gpu_type_ids) if gpu]
        if not gpus:
            raise CloudError("Indica al menos una GPU aceptable", 400)
        launch = {key: value for key, value in launch.items() if key != "gpu_type_id"}
        async with self._sessions() as session:
            row = CloudLaunchRequest(
                id=uuid.uuid4(),
                status="waiting",
                gpu_type_ids=gpus,
                launch=launch,
                retry_interval_s=max(30, int(retry_interval_s)),
                expires_at=_now() + timedelta(minutes=max(1, int(wait_minutes))),
                purpose=purpose,
                benchmark_run_id=benchmark_run_id,
            )
            session.add(row)
            await session.commit()
            request_id = row.id
        if try_now:
            await self._attempt(request_id)
        return await self.get(str(request_id))

    async def get(self, request_id: str) -> dict[str, Any]:
        from ocabra.db.cloud import CloudLaunchRequest

        async with self._sessions() as session:
            row = await session.get(CloudLaunchRequest, uuid.UUID(request_id))
            if row is None:
                raise CloudError("Petición no encontrada", 404)
            return request_out(row)

    async def list(self, *, include_finished: bool = True, limit: int = 50) -> list[dict[str, Any]]:
        from ocabra.db.cloud import CloudLaunchRequest

        query = (
            select(CloudLaunchRequest).order_by(CloudLaunchRequest.created_at.desc()).limit(limit)
        )
        if not include_finished:
            query = query.where(CloudLaunchRequest.status.in_(ACTIVE))
        async with self._sessions() as session:
            return [request_out(row) for row in (await session.execute(query)).scalars()]

    async def cancel(self, request_id: str) -> dict[str, Any]:
        from ocabra.db.cloud import CloudLaunchRequest

        async with self._sessions() as session:
            row = await session.get(CloudLaunchRequest, uuid.UUID(request_id))
            if row is None:
                raise CloudError("Petición no encontrada", 404)
            if row.status in ACTIVE:
                row.status = "cancelled"
                row.finished_at = _now()
                await session.commit()
            return request_out(row)

    async def _stamp_requested(self, node_id: str, requested_at, attempts: int) -> None:
        """El pod salió de una petición: su línea de tiempo empieza al pedirlo."""
        from ocabra.db.cloud import CloudNode

        async with self._sessions() as session:
            node = await session.get(CloudNode, uuid.UUID(node_id))
            if node is not None and requested_at is not None:
                timeline = dict(node.timeline or {})
                timeline["requested_at"] = requested_at.astimezone(UTC).isoformat()
                timeline["request_attempts"] = attempts
                node.timeline = timeline
                await session.commit()

    async def drive(self) -> None:
        """Una pasada: caduca las vencidas y prueba las que toca."""
        from ocabra.db.cloud import CloudLaunchRequest

        now = _now()
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(CloudLaunchRequest).where(CloudLaunchRequest.status.in_(ACTIVE))
                    )
                )
                .scalars()
                .all()
            )
            due: list[uuid.UUID] = []
            for row in rows:
                if row.expires_at <= now:
                    row.status = "expired"
                    row.finished_at = now
                    row.last_error = (
                        row.last_error or "no hubo capacidad en el plazo"
                    ) + " (caducada)"
                    continue
                last = row.last_attempt_at
                if last is None or (now - last).total_seconds() >= row.retry_interval_s:
                    due.append(row.id)
            await session.commit()
        for request_id in due:
            await self._attempt(request_id)

    async def _launch_gpu(self, gpu: str, launch: dict[str, Any]) -> dict[str, Any]:
        if launch.get("cloud_type") != "BOTH":
            return await self._cloud.launch(**launch, gpu_type_id=gpu)
        if launch.get("volume_id") or launch.get("privacy_mode") == "zero_data":
            return await self._cloud.launch(**{**launch, "cloud_type": "SECURE"}, gpu_type_id=gpu)

        catalog = await self._cloud.gpu_types()
        offer = next((item for item in catalog if item["id"] == gpu), {})
        for market in ("community", "secure"):
            if not offer.get(market) or offer.get(f"price_{market}") is None:
                continue
            try:
                return await self._cloud.launch(
                    **{**launch, "cloud_type": market.upper()}, gpu_type_id=gpu
                )
            except CloudCapacityError:
                continue
        raise CloudCapacityError(f"Sin capacidad en Community ni Secure para {gpu}")

    async def _attempt(self, request_id: uuid.UUID) -> None:
        """Prueba cada GPU aceptable, por orden, hasta lanzar o agotarlas."""
        from ocabra.db.cloud import CloudLaunchRequest

        async with self._sessions() as session:
            row = await session.get(CloudLaunchRequest, request_id)
            if row is None or row.status not in ACTIVE:
                return
            row.attempts += 1
            row.last_attempt_at = _now()
            gpus = list(row.gpu_type_ids or [])
            launch = dict(row.launch or {})
            await session.commit()

        result: dict[str, Any] | None = None
        launched_gpu: str | None = None
        errors: list[str] = []
        fatal: str | None = None
        for gpu in gpus:
            try:
                result = await self._launch_gpu(gpu, launch)
                launched_gpu = gpu
                break
            except CloudCapacityError as exc:
                errors.append(f"{gpu}: sin capacidad")
                logger.info(
                    "cloud_request_no_capacity",
                    request_id=str(request_id),
                    gpu=gpu,
                    error=str(exc)[:200],
                )
            except CloudError as exc:
                fatal = str(exc)
                break
            except Exception as exc:  # noqa: BLE001 — transitorio: se reintenta en la siguiente pasada
                errors.append(f"{gpu}: {type(exc).__name__}: {exc}"[:200])

        async with self._sessions() as session:
            row = await session.get(CloudLaunchRequest, request_id)
            if row is None or row.status not in ACTIVE:
                # Cancelada mientras se lanzaba: el pod, si salió, no se deja huérfano.
                if result is not None:
                    await self._cloud.terminate(result["id"], reason="request_cancelled")
                return
            if result is not None:
                await self._stamp_requested(result["id"], row.created_at, row.attempts)
                row.status = "launched"
                row.cloud_node_id = uuid.UUID(result["id"])
                row.launched_gpu_type_id = launched_gpu
                row.finished_at = _now()
                row.last_error = None
                logger.info("cloud_request_launched", request_id=str(request_id), gpu=launched_gpu)
            elif fatal is not None:
                row.status = "failed"
                row.finished_at = _now()
                row.last_error = fatal
            else:
                row.last_error = "; ".join(errors)[:2000] or None
            await session.commit()
