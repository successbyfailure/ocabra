"""Launch defaults for a model that has just been registered.

Registering a model leaves ``extra_config`` empty whenever the caller does not
send one, and installing from Explore never does. That was harmless while
Ollama served the weights — it decides GPU layers and context by itself — but
since 0.47.0 an Ollama pull *also* registers a ``llamacpp_mks`` deployment,
and llama.cpp takes only what it is told. With nothing set it falls back to
``settings.llama_cpp_gpu_layers`` (0, i.e. CPU only) and
``settings.llama_cpp_ctx_size`` (4096), so a 20 GiB model tries to run on the
CPU and never loads.

Reproduced 2026-11-09: ``ornith-1.5:35b``, ``flint:40b`` and
``nemotron-3.5-lightning:30b`` were installed from Explore and all three
landed with ``extra_config = NULL``.

Only missing keys are filled: anything the caller set wins, so this never
fights an explicit configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from ocabra.core import artifacts
from ocabra.registry.local_scanner import parse_gguf_context_length

logger = structlog.get_logger(__name__)

# Same shape every llama.cpp model in production runs with.
_DEFAULTS: dict[str, Any] = {
    "gpu_layers": 99,
    "flash_attn": True,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "kv_unified": True,
    "parallel_slots": 4,
    "cont_batching": True,
    # 60 s (the global default) is not enough for a 20 GiB model on a cold
    # page cache.
    "startup_timeout_s": 300,
}

# Ceiling for the context taken from the GGUF. Models now declare windows of
# a million tokens (``nemotron-3.5-lightning:30b``) that no card here can
# hold; the runtime presets are where a deployment goes above this.
_MAX_CTX = 262144

# When the file cannot be read: still usable, unlike 4096.
_FALLBACK_CTX = 32768


def llama_cpp_launch_defaults(
    extra_config: dict[str, Any] | None,
    *,
    artifact: dict[str, Any] | None = None,
    ollama_root: str | Path | None = None,
) -> dict[str, Any]:
    """Complete the ``llama_cpp`` section of *extra_config* with usable values.

    Args:
        extra_config: What the caller asked for; returned enriched, never
            overridden.
        artifact: Used to find the weights (for the native context length)
            and to tell whether the tag ships a vision projector.
        ollama_root: Override for the Ollama store, for tests.

    Returns:
        A new dict. ``ctx_size`` comes from the GGUF, capped at
        ``_MAX_CTX``; ``vision_mode`` is set to ``on_demand`` when a
        projector is present, so it is loaded on the first image instead of
        sitting in VRAM for every text request.
    """

    base = dict(extra_config or {})
    seccion = base.get(artifacts.config_section_for("llamacpp_mks"))
    seccion = dict(seccion) if isinstance(seccion, dict) else {}

    rutas: dict[str, str] = {}
    if artifact is not None:
        try:
            if ollama_root is None:
                rutas = artifacts.resolve_paths(artifact)
            else:
                rutas = artifacts.resolve_paths(artifact, ollama_root=ollama_root)
        except Exception as exc:  # la resolucion no puede tumbar un alta
            logger.warning("registration_defaults_artifact_unresolved", error=str(exc))

    for clave, valor in _DEFAULTS.items():
        seccion.setdefault(clave, valor)

    if "ctx_size" not in seccion:
        nativo = None
        ruta = rutas.get("model") or seccion.get("model_path")
        if ruta:
            nativo = parse_gguf_context_length(Path(str(ruta)))
        seccion["ctx_size"] = min(nativo, _MAX_CTX) if nativo else _FALLBACK_CTX

    if "vision_mode" not in seccion and rutas.get("mmproj"):
        seccion["vision_mode"] = "on_demand"

    base[artifacts.config_section_for("llamacpp_mks")] = seccion
    return base
