from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ocabra.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> JSONResponse:
    """Quick liveness check. Always returns 200 if the process is running."""
    return JSONResponse({"status": "ok", "version": settings.app_version})


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """
    Readiness check. Returns 200 only when all dependencies are healthy.

    Checks: postgres, redis, gpu_manager. Also reports count of loaded models.
    """
    from ocabra.redis_client import get_redis

    checks: dict[str, object] = {}

    # Check Redis
    try:
        await get_redis().ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    # Check Postgres (via SQLAlchemy engine ping)
    try:
        from ocabra.database import engine
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"

    # Check GPU Manager
    try:
        gpu_manager = request.app.state.gpu_manager
        if gpu_manager and gpu_manager._running:
            checks["gpu_manager"] = "ok"
        else:
            checks["gpu_manager"] = "error: not running"
    except Exception as e:
        checks["gpu_manager"] = f"error: {e}"

    # Count loaded models
    try:
        from ocabra.core.model_manager import ModelStatus
        model_manager = request.app.state.model_manager
        states = await model_manager.list_states()
        checks["models_loaded"] = sum(
            1 for s in states if s.status == ModelStatus.LOADED
        )
    except Exception:
        checks["models_loaded"] = 0

    string_checks = {k: v for k, v in checks.items() if isinstance(v, str)}
    all_ok = all(v == "ok" for v in string_checks.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        {"status": "ready" if all_ok else "degraded", "checks": checks},
        status_code=status_code,
    )
