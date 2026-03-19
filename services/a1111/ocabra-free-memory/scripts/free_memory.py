"""
oCabra extension – exposes POST /free-memory to flush PyTorch's CUDA cache.
Called by oCabra's ServiceManager after unload-checkpoint so that VRAM is
actually returned to the OS instead of staying in PyTorch's page cache.
"""
import gc

import torch
from modules import script_callbacks


def _flush():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def on_app_started(_demo, app):
    @app.post("/free-memory")
    async def free_memory():
        _flush()
        return {"status": "ok"}


script_callbacks.on_app_started(on_app_started)
