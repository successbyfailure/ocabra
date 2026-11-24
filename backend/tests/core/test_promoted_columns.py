"""REF-20: columnas ascendidas desde el cajón de sastre ``extra_config``.

Fijan la lógica del único puente
(``model_manager_helpers.promoted_columns_from_extra_config``) entre
``extra_config`` y las columnas reales de ``ModelConfig``, y que su mapeo de
secciones de override de VRAM no se desincroniza de ``config_schema``.
"""

from __future__ import annotations

from ocabra.core.config_schema import _VRAM_OVERRIDE_SECTIONS as SCHEMA_SECTIONS
from ocabra.core.config_schema import section_for
from ocabra.core.model_manager_helpers import (
    _VRAM_OVERRIDE_SECTIONS as HELPER_SECTIONS,
)
from ocabra.core.model_manager_helpers import (
    promoted_columns_from_extra_config,
)


def test_promueve_las_cuatro_desde_el_nivel_raiz_y_la_seccion():
    extra = {
        "_runtime": {"gpu_indices": [1], "parallelism": 4},
        "defaultReasoningEffort": "High",  # camelCase del formulario
        "llama_cpp": {"ctx_size": 65536, "vram_estimate_mb": 23000},
    }
    cols = promoted_columns_from_extra_config(extra, "llama_cpp")
    assert cols == {
        "runtime_state": {"gpu_indices": [1], "parallelism": 4},
        "default_reasoning_effort": "high",
        "context_length": 65536,
        "vram_estimate_mb": 23000,
    }


def test_context_length_prefiere_el_nivel_raiz():
    extra = {"context_length": 8192, "vllm": {"max_model_len": 4096}}
    assert promoted_columns_from_extra_config(extra, "vllm")["context_length"] == 8192


def test_vram_override_por_backend_vllm():
    extra = {"vllm": {"vram_estimate_mb": 15000}}
    assert promoted_columns_from_extra_config(extra, "vllm")["vram_estimate_mb"] == 15000


def test_sin_override_de_un_backend_sin_seccion_de_vram():
    # whisper no tiene sección de override de VRAM: no se inventa ninguna.
    extra = {"whisper": {"vram_estimate_mb": 9000}}
    assert promoted_columns_from_extra_config(extra, "whisper")["vram_estimate_mb"] is None


def test_todo_none_cuando_no_hay_nada_ni_con_extra_config_invalido():
    for extra in (None, {}, [], "x", {"_runtime": "no-dict", "default_reasoning_effort": ""}):
        cols = promoted_columns_from_extra_config(extra, "llama_cpp")
        assert cols == {
            "runtime_state": None,
            "default_reasoning_effort": None,
            "context_length": None,
            "vram_estimate_mb": None,
        }


def test_runtime_state_solo_acepta_dict():
    assert promoted_columns_from_extra_config({"_runtime": [1, 2]}, "llama_cpp")["runtime_state"] is (
        None
    )


def test_vram_override_descarta_booleanos_y_no_enteros():
    extra = {"llama_cpp": {"vram_estimate_mb": True}}
    assert promoted_columns_from_extra_config(extra, "llama_cpp")["vram_estimate_mb"] is None


def test_mapeo_de_secciones_de_vram_no_se_desincroniza_de_config_schema():
    # El helper es auto-contenido (lo copia la migración 0055); si config_schema
    # cambia sus secciones de override, este test obliga a actualizar el helper.
    assert HELPER_SECTIONS == {
        backend: section_for(backend) for backend in SCHEMA_SECTIONS
    }
