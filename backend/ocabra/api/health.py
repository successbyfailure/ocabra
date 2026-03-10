from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ocabra.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> JSONResponse:
    """Quick liveness check. Always returns 200 if the process is running."""
    return JSONResponse({"status": "ok", "version": settings.app_version})


@router.get("/ready")
async def ready() -> JSONResponse:
    """Readiness check. Returns 200 only when all dependencies are healthy."""
    from ocabra.redis_client import get_redis

    checks: dict[str, str] = {}

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

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        {"status": "ready" if all_ok else "degraded", "checks": checks},
        status_code=status_code,
    )
