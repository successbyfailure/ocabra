"""El tope de salida declarado se inyecta cuando el cliente no fija ninguno.

``max_output_tokens`` solo se usaba para anunciarlo en ``/v1/models`` y para
rechazar peticiones que pidiesen mas; nunca llegaba al backend. Una peticion
sin ``max_tokens`` generaba sin limite, asi que el tope era decorativo.

Incidente del 2026-11-07: gemma4:26b, con el tope en 65536, recibio una
peticion sin ``max_tokens``, se desboco y genero 132.263 tokens durante 26
minutos hasta que el timeout de inferencia la corto con un 504.
"""

from types import SimpleNamespace

import pytest

from ocabra.core.output_limits import apply_declared_output_limit

TOPE = 65536


def _perfil(max_output_tokens=TOPE):
    politica = {} if max_output_tokens is None else {"max_output_tokens": max_output_tokens}
    return SimpleNamespace(runtime_policy=politica)


# ── OpenAI ───────────────────────────────────────────────────────────────────


def test_inyecta_el_tope_cuando_el_cliente_no_pone_ninguno():
    body = {"messages": []}
    assert apply_declared_output_limit(_perfil(), body) == TOPE
    assert body["max_tokens"] == TOPE


@pytest.mark.parametrize("campo", ["max_tokens", "max_completion_tokens"])
def test_respeta_el_limite_que_pide_el_cliente(campo):
    body = {"messages": [], campo: 500}
    assert apply_declared_output_limit(_perfil(), body) is None
    assert body[campo] == 500
    assert body.get("max_tokens", 500) == 500


def test_un_cero_explicito_del_cliente_tambien_se_respeta():
    # 0 es un valor deliberado (y distinto de "no lo he puesto"); no se pisa.
    body = {"messages": [], "max_tokens": 0}
    assert apply_declared_output_limit(_perfil(), body) is None
    assert body["max_tokens"] == 0


def test_sin_tope_declarado_no_toca_nada():
    body = {"messages": []}
    assert apply_declared_output_limit(_perfil(None), body) is None
    assert "max_tokens" not in body


def test_un_tope_no_positivo_no_cuenta_como_tope():
    body = {"messages": []}
    assert apply_declared_output_limit(_perfil(0), body) is None
    assert "max_tokens" not in body


def test_un_perfil_sin_runtime_policy_no_rompe():
    body = {"messages": []}
    assert apply_declared_output_limit(SimpleNamespace(runtime_policy=None), body) is None
    assert "max_tokens" not in body


# ── Ollama nativo ────────────────────────────────────────────────────────────


def test_ollama_nativo_usa_options_num_predict():
    body = {"messages": []}
    assert apply_declared_output_limit(_perfil(), body, ollama_native=True) == TOPE
    assert body["options"] == {"num_predict": TOPE}
    assert "max_tokens" not in body


def test_ollama_nativo_conserva_las_opciones_que_ya_venian():
    body = {"messages": [], "options": {"temperature": 0.2}}
    assert apply_declared_output_limit(_perfil(), body, ollama_native=True) == TOPE
    assert body["options"] == {"temperature": 0.2, "num_predict": TOPE}


def test_ollama_nativo_respeta_el_num_predict_del_cliente():
    body = {"messages": [], "options": {"num_predict": 128}}
    assert apply_declared_output_limit(_perfil(), body, ollama_native=True) is None
    assert body["options"]["num_predict"] == 128


def test_ollama_nativo_respeta_un_num_predict_de_menos_uno():
    # -1 es "sin limite" en Ollama: es una eleccion explicita del cliente.
    body = {"messages": [], "options": {"num_predict": -1}}
    assert apply_declared_output_limit(_perfil(), body, ollama_native=True) is None
    assert body["options"]["num_predict"] == -1


# ── Acotado por el contexto del preset ───────────────────────────────────────
# Varios perfiles declaran 65536 de salida y tienen presets de 32k o 40960, asi
# que el tope declarado supera la ventana entera. Mientras no se inyectaba era
# inocuo; inyectandolo sin acotar se pediria mas salida que contexto hay.


def test_el_tope_se_acota_al_contexto_del_preset():
    body = {"messages": []}
    # Caso real: qwen3.8:27b-ollama, preset de 32k, tope declarado de 64k.
    assert apply_declared_output_limit(_perfil(65536), body, context_length=32768) == 32768
    assert body["max_tokens"] == 32768


def test_si_el_contexto_es_mayor_manda_el_tope_declarado():
    body = {"messages": []}
    assert apply_declared_output_limit(_perfil(16384), body, context_length=262144) == 16384


def test_sin_contexto_conocido_se_usa_el_tope_tal_cual():
    # context_length=0 significa "no lo se", no "cero".
    body = {"messages": []}
    assert apply_declared_output_limit(_perfil(TOPE), body, context_length=0) == TOPE


def test_un_contexto_negativo_o_raro_no_rompe():
    body = {"messages": []}
    assert apply_declared_output_limit(_perfil(TOPE), body, context_length=-5) == TOPE


def test_ollama_nativo_tambien_se_acota():
    body = {"messages": []}
    assert (
        apply_declared_output_limit(_perfil(65536), body, ollama_native=True, context_length=32768)
        == 32768
    )
    assert body["options"]["num_predict"] == 32768


# ── Respaldo global cuando el perfil no declara tope ─────────────────────────
# Nueve perfiles generativos no declaran ``max_output_tokens`` (qwen3:32b,
# qwen3:8b, mistral:7b, nemotron-3-nano:30b, bonsai y sus gemelos de ollama),
# asi que sin respaldo seguian generando sin limite.


def test_el_respaldo_global_cubre_a_los_perfiles_sin_tope():
    body = {"messages": []}
    assert apply_declared_output_limit(_perfil(None), body, fallback_limit=65536) == 65536
    assert body["max_tokens"] == 65536


def test_el_tope_del_perfil_manda_sobre_el_respaldo():
    body = {"messages": []}
    assert apply_declared_output_limit(_perfil(8192), body, fallback_limit=65536) == 8192


def test_el_respaldo_tambien_se_acota_al_contexto():
    body = {"messages": []}
    assert (
        apply_declared_output_limit(_perfil(None), body, fallback_limit=65536, context_length=32768)
        == 32768
    )


def test_un_respaldo_en_cero_significa_sin_respaldo():
    body = {"messages": []}
    assert apply_declared_output_limit(_perfil(None), body, fallback_limit=0) is None
    assert "max_tokens" not in body


def test_el_respaldo_no_pisa_el_limite_del_cliente():
    body = {"messages": [], "max_tokens": 42}
    assert apply_declared_output_limit(_perfil(None), body, fallback_limit=65536) is None
    assert body["max_tokens"] == 42


def test_el_respaldo_no_se_usa_para_rechazar():
    """``resolve_max_output_tokens`` es lo que rechaza peticiones que piden mas,
    y sigue devolviendo None sin tope de perfil: un cliente que pida 200k a un
    modelo sin tope declarado se sigue aceptando, no se empieza a devolver 400
    a quien hoy funciona."""
    from ocabra.core.output_limits import resolve_max_output_tokens

    assert resolve_max_output_tokens(_perfil(None)) is None
