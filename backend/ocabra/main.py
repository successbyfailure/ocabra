import asyncio
import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ocabra.config import settings

# Configure structlog early
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(settings.log_level)
    ),
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    logger.info("starting_ocabra", version=settings.app_version)

    from ocabra.redis_client import init_redis
    await init_redis()
    logger.info("redis_connected")

    # GPU Manager and Model Manager are initialized here when implemented
    # (Streams 1-A and 1-B). Stubs for now.

    logger.info("ocabra_ready")
    yield

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("shutting_down_ocabra")

    from ocabra.redis_client import close_redis
    await close_redis()

    logger.info("ocabra_stopped")


app = FastAPI(
    title="oCabra",
    description="Multi-GPU AI model server — OpenAI & Ollama compatible",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ROUTERS ──────────────────────────────────────────────────
# Each stream adds its router here. Do not remove this comment.
from ocabra.api.health import router as health_router  # noqa: E402

app.include_router(health_router)

# Stream 1-A: GPU Manager
# from ocabra.api.internal.gpus import router as gpus_router
# app.include_router(gpus_router, prefix="/ocabra")

# Stream 1-B: Model Manager
# from ocabra.api.internal.models import router as models_router
# app.include_router(models_router, prefix="/ocabra")

# Stream 1-C: Registry + Downloads
from ocabra.api.internal.downloads import router as downloads_router  # noqa: E402
from ocabra.api.internal.registry import router as registry_router  # noqa: E402

app.include_router(registry_router, prefix="/ocabra")
app.include_router(downloads_router, prefix="/ocabra")

# Stream 3-A: OpenAI API
# from ocabra.api.openai import router as openai_router
# app.include_router(openai_router, prefix="/v1")

# Stream 3-B: Ollama API
# from ocabra.api.ollama import router as ollama_router
# app.include_router(ollama_router)

# Stream 5: Metrics
# from ocabra.api.metrics import router as metrics_router
# app.include_router(metrics_router)
# ─────────────────────────────────────────────────────────────
