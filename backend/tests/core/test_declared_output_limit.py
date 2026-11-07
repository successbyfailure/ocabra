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
