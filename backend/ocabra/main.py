import logging
import os
import sys

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ocabra.config import settings

# ── Structlog configuration ───────────────────────────────────
_is_prod = os.getenv("OCABRA_ENV", "development").lower() == "production"

_processors = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
]

if _is_prod:
    _processors.append(structlog.processors.JSONRenderer())
else:
    _processors.append(structlog.dev.ConsoleRenderer(colors=True))

structlog.configure(
    processors=_processors,
    wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.log_level)),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def _build_cors_config() -> dict[str, object]:
    """Restrict browser access to same-origin production and loopback dev origins."""
    return {
        "allow_origin_regex": r"^https?://(localhost|127\.0\.0\.1)(?::\d+)?$",
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }


# The lifespan (startup/shutdown wiring, BUG-25/26) and the periodic loops live
# in bootstrap.py / maintenance_loops.py; the /docs prose and OpenAPI
# customization in openapi_docs.py. Re-exported here so tests and other modules
# can keep importing them from ocabra.main.
from ocabra.bootstrap import lifespan  # noqa: E402
from ocabra.openapi_docs import _API_DESCRIPTION, build_openapi_schema  # noqa: E402

app = FastAPI(
    title="oCabra",
    description=_API_DESCRIPTION,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


def _custom_openapi():
    """Enrich the auto-generated OpenAPI spec with typed request schemas."""
    return build_openapi_schema(app)


app.openapi = _custom_openapi

app.add_middleware(
    CORSMiddleware,
    **_build_cors_config(),
)


# Global 500 logger: FastAPI's default exception handler returns a bare
# "Internal Server Error" with no traceback in the logs, which makes
# diagnosing federation/proxy issues painful. Log the full traceback for
# every uncaught exception so it's never silent.
from ocabra.api._errors import error_content, error_family, request_path  # noqa: E402
from ocabra.core.worker_pool import VisionNoRoomError  # noqa: E402


@app.exception_handler(VisionNoRoomError)
async def _vision_no_room_handler(request, exc):  # noqa: ANN001
    """An image a model cannot take right now (no room for its vision
    projector): 503 ``vision_no_room`` instead of a bare 500."""
    from fastapi.responses import JSONResponse

    logger.info("vision_no_room", path=request.url.path, error=str(exc))
    # Same envelope as every other oCabra error (``HTTPException`` detail),
    # plus the family's top-level ``error`` (REF-10).
    detail = {
        "error": {
            "message": str(exc),
            "type": "server_error",
            "code": "vision_no_room",
        }
    }
    return JSONResponse(status_code=503, content=error_content(request.url.path, 503, detail))


from starlette.exceptions import HTTPException as _StarletteHTTPException  # noqa: E402


@app.exception_handler(_StarletteHTTPException)
async def _peer_fallback_not_found_handler(request, exc):  # noqa: ANN001
    """A model only a peer has, whose peer failed: 503, not "does not exist".

    When the chosen peer refuses or is unreachable the request falls back to
    local, and a model that exists only on that peer (a pod's) then answered
    404 "model not found": wrong, and not retryable. The fallback is recorded
    in ``federation_info``; here a 404 after it becomes a retryable 503 that
    names the node.
    """
    from fastapi.exception_handlers import http_exception_handler
    from fastapi.responses import JSONResponse

    path = request_path(request)
    info = getattr(request.state, "federation_info", None) or {}
    if exc.status_code == 404 and info.get("fallback"):
        peer = info.get("fallback_peer") or "a federation peer"
        detail = info.get("fallback_detail")
        message = (
            f"The model is only available on node '{peer}', which did not serve the "
            f"request ({detail}). Retry shortly." if detail
            else f"The model is only available on node '{peer}', which did not serve "
            "the request. Retry shortly."
        )  # fmt: skip
        detail_body = {
            "error": {
                "message": message,
                "type": "server_error",
                "code": "peer_unavailable",
            }
        }
        return JSONResponse(
            status_code=503,
            content=error_content(path, 503, detail_body),
            headers={"Retry-After": "15"},
        )
    if error_family(path) is None:
        return await http_exception_handler(request, exc)
    # /v1 and /api (REF-10): FastAPI's handler plus the family's ``error``.
    from fastapi.utils import is_body_allowed_for_status_code
    from starlette.responses import Response

    headers = getattr(exc, "headers", None)
    if not is_body_allowed_for_status_code(exc.status_code):
        return Response(status_code=exc.status_code, headers=headers)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_content(path, exc.status_code, exc.detail),
        headers=headers,
    )


from fastapi.exceptions import RequestValidationError as _RequestValidationError  # noqa: E402


@app.exception_handler(_RequestValidationError)
async def _request_validation_handler(request, exc):  # noqa: ANN001
    """FastAPI's 422, plus the family's top-level ``error`` on /v1 and /api."""
    from fastapi.encoders import jsonable_encoder
    from fastapi.exception_handlers import request_validation_exception_handler
    from fastapi.responses import JSONResponse

    if error_family(request.url.path) is None:
        return await request_validation_exception_handler(request, exc)
    detail = jsonable_encoder(exc.errors())
    return JSONResponse(status_code=422, content=error_content(request.url.path, 422, detail))


@app.exception_handler(Exception)
async def _log_unhandled_exception(request, exc):
    import traceback as _tb

    from fastapi.responses import JSONResponse

    logger.warning(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__,
        traceback=_tb.format_exc(),
    )
    detail = {
        "error": {
            "message": "Internal Server Error",
            "type": "server_error",
            "code": type(exc).__name__,
        }
    }
    return JSONResponse(status_code=500, content=error_content(request.url.path, 500, detail))

# Stream 3-A: Stats middleware for /v1/* endpoints. Records only.
from ocabra.core.admission import AdmissionMiddleware  # noqa: E402
from ocabra.stats.collector import StatsMiddleware, record_admission_rejection  # noqa: E402

app.add_middleware(StatsMiddleware)
# Admission (in-flight cap, maintenance, circuit breaker; REF-08): just outside
# the stats, so its slot spans the whole response, and inside the federation
# hop marker, whose ``federation_info`` it reads. Its rejections never reach
# the stats middleware: they are recorded through ``on_reject``.
app.add_middleware(AdmissionMiddleware, on_reject=record_admission_rejection)
# Outermost: marks requests that came from a federation peer before anything
# else runs, so they are never forwarded to yet another peer.
from ocabra.benchmarks.sampling_override import BenchmarkSamplingMiddleware  # noqa: E402
from ocabra.core.federation import FederationHopMiddleware  # noqa: E402

app.add_middleware(FederationHopMiddleware)
app.add_middleware(BenchmarkSamplingMiddleware)

from ocabra.core.privacy import PrivacyHeaderMiddleware  # noqa: E402

# X-Ocabra-Privacy on every response: which privacy mode served it.
app.add_middleware(PrivacyHeaderMiddleware)

from ocabra.api._risky_config import RiskyConfigMiddleware  # noqa: E402

# Options that run arbitrary code (trust_remote_code, extra_args, …) are for
# system_admin only, however they get in (SEG-04).
app.add_middleware(RiskyConfigMiddleware)

# ── ROUTERS ──────────────────────────────────────────────────
# Each stream adds its router here. Do not remove this comment.
from ocabra.api.health import router as health_router  # noqa: E402

app.include_router(health_router)

# Paginas de documentacion en prosa (/docs/changelog, /docs/models). Se montan
# despues de las rutas de Swagger para que /docs y /docs/oauth2-redirect, que
# FastAPI registra al crear la app, sigan ganando.
from ocabra.api.docs_pages import router as docs_pages_router  # noqa: E402

app.include_router(docs_pages_router)

# ── Internal routers (hidden from /docs — admin dashboard only) ──
from ocabra.api.internal.downloads import router as downloads_router  # noqa: E402
from ocabra.api.internal.gpus import router as gpus_router  # noqa: E402
from ocabra.api.internal.host import router as host_router  # noqa: E402
from ocabra.api.internal.logical_models import router as logical_models_router  # noqa: E402
from ocabra.api.internal.models import router as models_router  # noqa: E402
from ocabra.api.internal.models_update import router as models_update_router  # noqa: E402
from ocabra.api.internal.ollama_admin import router as ollama_admin_router  # noqa: E402
from ocabra.api.internal.registry import router as registry_router  # noqa: E402
from ocabra.api.internal.services import router as services_router  # noqa: E402
from ocabra.api.internal.status import router as status_router  # noqa: E402
from ocabra.api.internal.system_update import router as system_update_router  # noqa: E402
from ocabra.api.internal.video import router as video_router  # noqa: E402
from ocabra.api.internal.ws import router as ws_router  # noqa: E402

app.include_router(gpus_router, prefix="/ocabra", include_in_schema=False)
app.include_router(status_router, prefix="/ocabra", include_in_schema=False)
app.include_router(video_router, prefix="/ocabra", include_in_schema=False)
# Profiles before models: models' catch-all GET /models/{model_id:path} would
# otherwise swallow GET /models/{id}/profiles (it answered 404 "model
# '<id>/profiles' not found").
from ocabra.api.internal.profiles import router as profiles_router  # noqa: E402

app.include_router(profiles_router, prefix="/ocabra", include_in_schema=False)
from ocabra.api.internal.openai_image_compat import (  # noqa: E402
    router as openai_image_compat_router,
)
from ocabra.api.internal.openai_tts_compat import router as openai_tts_compat_router  # noqa: E402

app.include_router(openai_tts_compat_router, prefix="/ocabra", include_in_schema=False)
app.include_router(openai_image_compat_router, prefix="/ocabra", include_in_schema=False)
app.include_router(models_router, prefix="/ocabra", include_in_schema=False)
app.include_router(system_update_router, prefix="/ocabra", include_in_schema=False)
app.include_router(logical_models_router, prefix="/ocabra", include_in_schema=False)
app.include_router(models_update_router, prefix="/ocabra", include_in_schema=False)
app.include_router(ws_router, prefix="/ocabra", include_in_schema=False)
app.include_router(registry_router, prefix="/ocabra", include_in_schema=False)
app.include_router(downloads_router, prefix="/ocabra", include_in_schema=False)
app.include_router(services_router, prefix="/ocabra", include_in_schema=False)
app.include_router(host_router, prefix="/ocabra", include_in_schema=False)
app.include_router(ollama_admin_router, prefix="/ocabra", include_in_schema=False)

# Bloque 15 — Modular backends router
from ocabra.api.internal.backends import router as backends_router  # noqa: E402

app.include_router(backends_router, prefix="/ocabra", include_in_schema=False)
from ocabra.api.internal.federation import router as federation_router  # noqa: E402

app.include_router(federation_router, prefix="/ocabra", include_in_schema=False)
from ocabra.api.internal.cloud import router as cloud_router  # noqa: E402

app.include_router(cloud_router, prefix="/ocabra", include_in_schema=False)

# Stream 3-A: OpenAI API
from ocabra.api.openai import router as openai_router  # noqa: E402

app.include_router(openai_router, prefix="/v1")

# Stream 3-B: Ollama API
from ocabra.api.ollama import router as ollama_router  # noqa: E402

app.include_router(ollama_router)

# Stream 5: Metrics (public), Config/Stats/TRT/Auth/Users/Groups/Profiles (internal)
from ocabra.api.internal.auth import router as auth_router  # noqa: E402
from ocabra.api.internal.benchmarks import router as benchmarks_router  # noqa: E402
from ocabra.api.internal.config import router as config_router  # noqa: E402
from ocabra.api.internal.estimate import router as estimate_router  # noqa: E402
from ocabra.api.internal.groups import router as groups_router  # noqa: E402
from ocabra.api.internal.sessions import router as sessions_router  # noqa: E402
from ocabra.api.internal.stats import router as stats_router  # noqa: E402
from ocabra.api.internal.trtllm import router as trtllm_router  # noqa: E402
from ocabra.api.internal.users import router as users_router  # noqa: E402
from ocabra.api.metrics import router as metrics_router  # noqa: E402

app.include_router(metrics_router)
app.include_router(config_router, prefix="/ocabra", include_in_schema=False)
app.include_router(stats_router, prefix="/ocabra", include_in_schema=False)
app.include_router(trtllm_router, prefix="/ocabra", include_in_schema=False)
app.include_router(auth_router, prefix="/ocabra", include_in_schema=False)
app.include_router(benchmarks_router, prefix="/ocabra", include_in_schema=False)
app.include_router(users_router, prefix="/ocabra", include_in_schema=False)
app.include_router(groups_router, prefix="/ocabra", include_in_schema=False)
app.include_router(estimate_router, prefix="/ocabra", include_in_schema=False)
app.include_router(sessions_router, prefix="/ocabra", include_in_schema=False)

# Agents + MCP (plan: docs/tasks/agents-mcp-plan.md)
from ocabra.api.internal.agents import router as agents_router  # noqa: E402
from ocabra.api.internal.mcp_servers import router as mcp_servers_router  # noqa: E402

app.include_router(agents_router, prefix="/ocabra", include_in_schema=False)
app.include_router(mcp_servers_router, prefix="/ocabra", include_in_schema=False)

from ocabra.api.internal.stats_agents import router as stats_agents_router  # noqa: E402

app.include_router(stats_agents_router, prefix="/ocabra", include_in_schema=False)
# ─────────────────────────────────────────────────────────────
