from fastapi import APIRouter

from .chat import router as chat_router
from .delete import router as delete_router
from .embeddings import router as embeddings_router
from .generate import router as generate_router
from .pull import router as pull_router
from .show import router as show_router
from .tags import router as tags_router

router = APIRouter(prefix="/api", tags=["Ollama"])
router.include_router(tags_router)
router.include_router(show_router)
router.include_router(pull_router)
router.include_router(generate_router)
router.include_router(chat_router)
router.include_router(embeddings_router)
router.include_router(delete_router)
