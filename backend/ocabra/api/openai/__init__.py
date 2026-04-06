from fastapi import APIRouter

from .audio import router as audio_router
from .chat import router as chat_router
from .completions import router as completions_router
from .embeddings import router as embeddings_router
from .images import router as images_router
from .models import router as models_router
from .pooling import router as pooling_router
from .realtime import router as realtime_router

router = APIRouter(tags=["OpenAI"])
router.include_router(models_router)
router.include_router(chat_router)
router.include_router(completions_router)
router.include_router(embeddings_router)
router.include_router(pooling_router)
router.include_router(images_router)
router.include_router(audio_router)
router.include_router(realtime_router)
