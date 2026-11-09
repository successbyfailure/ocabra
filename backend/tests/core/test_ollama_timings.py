"""Las respuestas de Ollama tienen que llevar los tiempos reales del motor.

Ollama publica ``prompt_eval_duration`` y ``eval_duration`` en nanosegundos y
sus clientes calculan tokens/s dividiendo por ellos. oCabra ponia
``eval_duration = total_duration`` —el reloj de pared entero, con cola, carga
y prefill dentro— y ``prompt_eval_duration`` ni se mandaba. El cliente veia
una generacion mas lenta de la real y el prefill, invisible.

De paso, es lo que dejaba a ``/api/chat`` y ``/api/generate`` sin una sola
cifra de rendimiento guardada: 1.339 peticiones a ciegas en los 7 dias previos
al 2026-11-09, el 29% del trafico generativo.
"""

from __future__ import annotations

from ocabra.api.ollama.chat import _ollama_durations
from ocabra.stats.collector import _extract_speculative_stats


def test_los_tiempos_salen_de_las_timings_del_motor():
    total = 9_000_000_000  # 9 s de reloj de pared
    salida = _ollama_durations({"timings": {"prompt_ms": 1500.0, "predicted_ms": 4000.0}}, total)
    assert salida["prompt_eval_duration"] == 1_500_000_000
    assert salida["eval_duration"] == 4_000_000_000
    # El total sigue siendo el reloj de pared: ahi si entra la cola.
    assert salida["total_duration"] == total


def test_sin_timings_no_se_inventa_el_reparto():
    salida = _ollama_durations({}, 9_000_000_000)
    assert salida["eval_duration"] == 9_000_000_000
    assert "prompt_eval_duration" not in salida


def test_la_generacion_ya_no_se_lleva_el_tiempo_de_cola():
    """Con 9 s de reloj y 4 de generacion, 100 tokens son 25 t/s, no 11."""
    salida = _ollama_durations({"timings": {"predicted_ms": 4000.0}}, 9_000_000_000)
    tps = 100 / (salida["eval_duration"] / 1e9)
    assert round(tps) == 25


def test_el_recolector_entiende_la_forma_de_ollama():
    respuesta = {
        "prompt_eval_count": 500,
        "prompt_eval_duration": 1_000_000_000,
        "eval_count": 100,
        "eval_duration": 2_000_000_000,
    }
    draft_n, aceptados, generacion, prefill = _extract_speculative_stats(respuesta)
    assert generacion == 50.0
    assert prefill == 500.0
    # La especulativa solo la reporta llama.cpp en su propio bloque.
    assert draft_n is None and aceptados is None


def test_el_recolector_sigue_leyendo_la_forma_de_llama_cpp():
    respuesta = {
        "timings": {
            "predicted_per_second": 57.3,
            "prompt_per_second": 180.0,
            "draft_n": 4,
            "draft_n_accepted": 3,
        }
    }
    assert _extract_speculative_stats(respuesta) == (4, 3, 57.3, 180.0)


def test_una_duracion_a_cero_no_da_una_velocidad_infinita():
    respuesta = {
        "eval_count": 100,
        "eval_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 5,
    }
    _, _, generacion, prefill = _extract_speculative_stats(respuesta)
    assert generacion is None
    assert prefill is None
