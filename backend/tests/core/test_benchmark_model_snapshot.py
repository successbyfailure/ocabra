"""Foto del modelo con el que se midió (benchmarks/model_snapshot.py)."""

from __future__ import annotations

import pytest

from ocabra.benchmarks.model_snapshot import build_snapshot, capture

PROFILE = {
    "profile_id": "gemma4:26b",
    "base_model_id": "llamacpp_mks/gemma4:26b",
    "request_defaults": {"temperature": 1.0, "top_k": 64},
    "runtime_presets": [
        {"preset_key": "normal", "context_mode": "fixed", "context_tokens": 262144,
         "parallelism": 4, "kv_cache_dtype": "q8", "enabled": True, "is_default": True},
    ],
}  # fmt: skip
MODEL = {
    "model_id": "llamacpp_mks/gemma4:26b",
    "backend_type": "llamacpp_mks",
    "backend_model_id": "gemma4:26b",
    "extra_config": {
        "llama_cpp": {
            "ctx_size": 262144,
            "model_path": "/data/ollama_models/blobs/sha256-abc",
            "speculative": {"spec_type": "draft-mtp"},
        },
        "_runtime": {"secret": "no"},
    },
}


def test_la_foto_guarda_configuracion_presets_y_muestreo_sin_rutas():
    resolved = [{"preset_key": "normal", "effective_context": 262144, "fits": True}]
    snap = build_snapshot(PROFILE, MODEL, resolved)
    assert snap["requestDefaults"] == {"temperature": 1.0, "top_k": 64}
    assert snap["config"]["model_path"] == "sha256-abc"  # sin la ruta de aquí
    assert snap["config"]["speculative"] == {"spec_type": "draft-mtp"}
    assert snap["modelFile"] == "sha256-abc" and snap["backendModelId"] == "gemma4:26b"
    (preset,) = snap["presets"]
    assert preset["contextTokens"] == 262144 and preset["effectiveContext"] == 262144
    assert preset["parallelism"] == 4 and preset["fits"] is True


@pytest.mark.asyncio
async def test_captura_por_la_api_tambien_con_ids_de_despliegue():
    calls = []
    ollama = {"model_id": "ollama/gemma4:12b", "backend_type": "ollama",
              "backend_model_id": "gemma4:12b", "extra_config": {"registry_metadata": {}}}  # fmt: skip
    pages = {
        "/ocabra/models/ollama/gemma4:12b/profiles": [
            {"profile_id": "gemma4:12b-ollama", "base_model_id": "ollama/gemma4:12b",
             "is_default": True, "runtime_presets": []},
        ],
        "/ocabra/models/ollama/gemma4:12b": ollama,
        "/ocabra/config": {"ollamaNumParallel": 1, "ollamaMaxContext": 65536},
    }  # fmt: skip

    async def get_json(path):
        calls.append(path)
        return pages.get(path)

    snap = await capture(get_json, "ollama/gemma4:12b")
    assert snap["profileId"] == "gemma4:12b-ollama" and snap["backendType"] == "ollama"
    assert snap["config"] == {} and snap["backendModelId"] == "gemma4:12b"
    assert snap["server"] == {"ollamaNumParallel": 1, "ollamaMaxContext": 65536}
    assert await capture(lambda path: _none(), "no-existe") is None


async def _none():
    return None
