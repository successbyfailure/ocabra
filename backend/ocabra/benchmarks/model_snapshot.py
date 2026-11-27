"""Foto del modelo con el que se midió: configuración, presets, muestreo y fichero.

Para comparar dos medidas hace falta saber con qué se hicieron: el contexto y
los slots (en el hub, gemma4:26b va con 262 144 tokens × 4; en un pod, con su
preset de 64k), cómo se reparte entre GPUs, el tipo de KV, el MTP, la
cuantización y el muestreo por defecto del perfil. Se pide a la máquina donde
se midió (el hub o un nodo de la federación) por su API, al terminar el
intento, y se guarda en ``environment.model.snapshot``.

No guarda rutas de este servidor: de una ruta queda solo el nombre del fichero.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

GetJson = Callable[[str], Awaitable[Any]]

# Sección de ``extra_config`` que lee cada backend.
_SECTION = {"llamacpp_mks": "llama_cpp", "llama_cpp": "llama_cpp"}
# Lo que no es configuración de carga.
_NOT_CONFIG = frozenset({"registry_metadata"})
_PRESET_FIELDS = (
    ("preset_key", "key"),
    ("context_mode", "contextMode"),
    ("context_tokens", "contextTokens"),
    ("parallelism", "parallelism"),
    ("kv_cache_dtype", "kvCacheDtype"),
    ("placement_policy", "placementPolicy"),
    ("enabled", "enabled"),
    ("is_default", "default"),
)


def _field(obj: dict, snake: str) -> Any:
    if snake in obj:
        return obj[snake]
    head, *rest = snake.split("_")
    return obj.get(head + "".join(part.title() for part in rest))


def _strip_paths(value: Any) -> Any:
    """Las rutas absolutas pasan a ser solo el nombre del fichero."""
    if isinstance(value, dict):
        return {k: _strip_paths(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [_strip_paths(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        return os.path.basename(value.rstrip("/")) or value
    return value


def build_snapshot(
    profile: dict,
    model: dict | None,
    resolved: list | None,
    *,
    source: str = "live",
    server: dict | None = None,
) -> dict[str, Any]:
    """Compone la foto a partir de lo que devuelve la API (puro, sin red)."""
    model = model or {}
    backend = str(_field(model, "backend_type") or _field(profile, "backend") or "")
    extra = _field(model, "extra_config") or {}
    extra = extra if isinstance(extra, dict) else {}
    section_name = _SECTION.get(backend, backend)
    section = extra.get(section_name) if isinstance(extra.get(section_name), dict) else None
    if section is None:
        section_free = {k: v for k, v in extra.items() if k not in _NOT_CONFIG}
        config = _strip_paths(section_free)
    else:
        config = _strip_paths(section)

    by_key = {}
    for row in resolved or []:
        if isinstance(row, dict) and _field(row, "preset_key"):
            by_key[_field(row, "preset_key")] = row
    presets = []
    for preset in _field(profile, "runtime_presets") or []:
        if not isinstance(preset, dict):
            continue
        item = {out: _field(preset, key) for key, out in _PRESET_FIELDS}
        live = by_key.get(item["key"]) or {}
        if live:
            item["effectiveContext"] = _field(live, "effective_context")
            item["effectiveParallelism"] = _field(live, "effective_parallelism")
            item["fits"] = _field(live, "fits")
            item["estimatedVramMb"] = _field(live, "estimated_vram_mb")
        presets.append(item)

    artifact = _field(model, "artifact") or {}
    artifact = artifact if isinstance(artifact, dict) else {}
    model_path = (section or {}).get("model_path") if section else None
    return {
        "source": source,
        "capturedAt": datetime.now(UTC).isoformat(),
        "profileId": _field(profile, "profile_id"),
        "baseModelId": _field(profile, "base_model_id") or _field(model, "model_id"),
        "backendType": backend or None,
        "requestDefaults": _field(profile, "request_defaults"),
        "loadOverrides": _strip_paths(_field(profile, "load_overrides")),
        "presets": presets,
        "config": config,
        "artifact": {
            k: _strip_paths(artifact.get(k))
            for k in ("kind", "source", "ref", "file")
            if k in artifact
        }
        or None,
        "modelFile": os.path.basename(str(model_path)) if model_path else None,
        # El tag de Ollama o el id con el que el backend conoce el modelo.
        "backendModelId": _field(model, "backend_model_id"),
        # Ollama: un daemon con paralelismo y contexto globales (Ajustes).
        "server": server or None,
    }


async def capture(get_json: GetJson, name: str, *, source: str = "live") -> dict[str, Any] | None:
    """La foto de *name* (un perfil o ``<backend>/<modelo>``) en una máquina.

    *get_json* hace un GET a la API de esa máquina y devuelve el JSON o ``None``.
    """
    profile = await get_json(f"/ocabra/profiles/{quote(name, safe=':')}")
    if not isinstance(profile, dict) and "/" in name:
        listed = await get_json(f"/ocabra/models/{quote(name, safe='/:')}/profiles")
        candidates = [p for p in listed or [] if isinstance(p, dict)]
        profile = next((p for p in candidates if _field(p, "is_default")), None) or (
            candidates[0] if candidates else None
        )
    if not isinstance(profile, dict):
        return None
    base = _field(profile, "base_model_id")
    model = await get_json(f"/ocabra/models/{quote(str(base), safe='/:')}") if base else None
    pid = quote(str(_field(profile, "profile_id")), safe=":")
    resolved = await get_json(f"/ocabra/profiles/{pid}/runtime-presets/resolved")
    server = None
    backend = str(_field(model, "backend_type") or "") if isinstance(model, dict) else ""
    if backend == "ollama":
        cfg = await get_json("/ocabra/config")
        if isinstance(cfg, dict):
            server = {
                "ollamaNumParallel": _field(cfg, "ollama_num_parallel"),
                "ollamaMaxContext": _field(cfg, "ollama_max_context"),
            }
    return build_snapshot(
        profile,
        model if isinstance(model, dict) else None,
        resolved if isinstance(resolved, list) else None,
        source=source,
        server=server,
    )
