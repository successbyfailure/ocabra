"""Una peticion suelta tiene que poder diagnosticarse sin abrir la base de datos.

Espera en cola, velocidad de lectura del prompt y velocidad de generacion son
tres cosas distintas: una cola larga es contencion, un prefill lento es un
prompt enorme, y una generacion lenta es el modelo o la tarjeta. Estaba todo
guardado pero el listado por peticion no lo publicaba, asi que para verlo habia
que ir al agregado por modelo —que promedia justo lo que quieres separar— o
consultar Postgres a mano.
"""

from __future__ import annotations

from types import SimpleNamespace

from ocabra.stats.aggregator import _rendimiento_de_la_peticion


def _fila(**kwargs):
    base = {
        "runtime_telemetry": {"queue_ms": 1200, "prefill_ms": 458},
        "prompt_tokens_per_second": 289.9,
        "predicted_tokens_per_second": 23.3,
        "generation_ms": 52139,
        "cached_tokens": 0,
        "effective_context": 262144,
        "runtime_preset": "extended",
        "gpu_index": 1,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_las_tres_cifras_que_se_diagnostican_por_separado():
    salida = _rendimiento_de_la_peticion(_fila())
    assert salida["queueMs"] == 1200
    assert salida["promptTokensPerSecond"] == 289.9
    assert salida["predictedTokensPerSecond"] == 23.3


def test_tambien_el_contexto_y_el_preset_con_que_corrio():
    salida = _rendimiento_de_la_peticion(_fila())
    assert salida["effectiveContext"] == 262144
    assert salida["runtimePreset"] == "extended"
    assert salida["gpuIndex"] == 1
    assert salida["generationMs"] == 52139


def test_sin_telemetria_no_se_inventa_una_espera():
    salida = _rendimiento_de_la_peticion(_fila(runtime_telemetry=None))
    assert salida["queueMs"] is None
    # Lo demas sigue saliendo: vive en columnas, no en el JSON.
    assert salida["predictedTokensPerSecond"] == 23.3


def test_una_espera_corrupta_no_rompe_el_listado():
    salida = _rendimiento_de_la_peticion(_fila(runtime_telemetry={"queue_ms": "pronto"}))
    assert salida["queueMs"] is None


def test_una_peticion_antigua_sin_esos_campos_no_falla():
    """Las filas guardadas antes de que existieran estas columnas."""
    vieja = SimpleNamespace(runtime_telemetry={})
    salida = _rendimiento_de_la_peticion(vieja)
    assert salida["promptTokensPerSecond"] is None
    assert salida["queueMs"] is None
