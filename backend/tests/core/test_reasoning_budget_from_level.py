"""El nivel de ``reasoning_effort`` se traduce a un tope duro en llama.cpp.

El nivel por si solo no limitaba nada: viajaba como ``reasoning_effort`` y como
``chat_template_kwargs.enable_thinking``, que son sugerencias que la plantilla
puede ignorar. La de gemma4 **no menciona** ``reasoning_effort`` (comprobado en
la plantilla que publica llama-server en ``/props``), asi que "low", "medium" y
"high" producian resultados identicos: 1178 caracteres de razonamiento en los
tres. El unico tope real es ``reasoning_budget_tokens``, y solo se enviaba si
el cliente lo pedia a mano.
"""

from types import SimpleNamespace

import pytest

from ocabra.api.openai._deps import _apply_reasoning_effort
from ocabra.config import settings


def _estado(backend="llamacpp_mks"):
    return SimpleNamespace(
        backend_type=backend,
        extra_config={},
        capabilities=SimpleNamespace(reasoning=True),
    )


@pytest.mark.parametrize(
    ("nivel", "esperado"),
    [
        ("low", settings.reasoning_budget_low_tokens),
        ("medium", settings.reasoning_budget_medium_tokens),
    ],
)
def test_cada_nivel_trae_su_tope_duro(nivel, esperado):
    payload = {"reasoning_effort": nivel}
    _apply_reasoning_effort(_estado(), payload)
    assert payload["reasoning_budget_tokens"] == esperado
    assert payload["chat_template_kwargs"]["enable_thinking"] is True


def test_los_niveles_dejan_de_ser_equivalentes():
    """La regresion concreta: low y medium daban exactamente lo mismo."""
    bajo, medio = {"reasoning_effort": "low"}, {"reasoning_effort": "medium"}
    _apply_reasoning_effort(_estado(), bajo)
    _apply_reasoning_effort(_estado(), medio)
    assert bajo["reasoning_budget_tokens"] < medio["reasoning_budget_tokens"]


def test_high_no_lleva_tope_por_defecto():
    # Quien pide esfuerzo alto pide justamente que piense sin prisa.
    payload = {"reasoning_effort": "high"}
    _apply_reasoning_effort(_estado(), payload)
    assert "reasoning_budget_tokens" not in payload


def test_off_no_necesita_tope_porque_apaga_el_pensamiento():
    payload = {"reasoning_effort": "off"}
    _apply_reasoning_effort(_estado(), payload)
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert "reasoning_budget_tokens" not in payload


def test_un_presupuesto_explicito_del_cliente_manda():
    payload = {"reasoning_effort": "low", "reasoning_budget": 50}
    _apply_reasoning_effort(_estado(), payload)
    # No se pisa: lo consume despues _apply_reasoning_budget.
    assert "reasoning_budget_tokens" not in payload
    assert payload["reasoning_budget"] == 50


def test_un_reasoning_budget_tokens_ya_puesto_no_se_pisa():
    payload = {"reasoning_effort": "low", "reasoning_budget_tokens": 7}
    _apply_reasoning_effort(_estado(), payload)
    assert payload["reasoning_budget_tokens"] == 7


@pytest.mark.parametrize("backend", ["ollama", "vllm", "whisper"])
def test_otros_backends_no_reciben_el_campo_de_llama_cpp(backend):
    """``reasoning_budget_tokens`` es nativo de llama-server; nadie mas lo entiende."""
    payload = {"reasoning_effort": "low"}
    _apply_reasoning_effort(_estado(backend), payload)
    assert "reasoning_budget_tokens" not in payload


def test_el_tope_es_configurable(monkeypatch):
    monkeypatch.setattr(settings, "reasoning_budget_low_tokens", 99)
    payload = {"reasoning_effort": "low"}
    _apply_reasoning_effort(_estado(), payload)
    assert payload["reasoning_budget_tokens"] == 99


def test_un_tope_en_cero_significa_sin_tope(monkeypatch):
    monkeypatch.setattr(settings, "reasoning_budget_low_tokens", 0)
    payload = {"reasoning_effort": "low"}
    _apply_reasoning_effort(_estado(), payload)
    assert "reasoning_budget_tokens" not in payload


# ── El nivel tiene que llegar a la plantilla ─────────────────────────────────
# llama-server NO reenvia a la plantilla los campos de primer nivel: comprobado
# en vivo contra gemma4:26b, que si lee ``enable_thinking``. Dentro de
# ``chat_template_kwargs`` funciona (0 caracteres de razonamiento); de primer
# nivel se ignora (295). Enviando el nivel solo arriba, las plantillas que si lo
# leen caian a su propio default — ``xhigh`` en Qwen 3.5 — asi que qwen3.8:27b,
# su abliterated y RavenX-Chaos-Agent recibian siempre el esfuerzo maximo.


def test_el_nivel_viaja_dentro_de_chat_template_kwargs():
    payload = {"reasoning_effort": "low"}
    _apply_reasoning_effort(_estado(), payload)
    assert payload["chat_template_kwargs"]["reasoning_effort"] == "low"


def test_high_se_traduce_a_xhigh():
    """Las plantillas de Qwen 3.5 validan ``not in ('xhigh','medium','low')`` y
    lanzan excepcion con cualquier otra cosa. Comprobado renderizando la
    plantilla real de qwen3.8-27b: ``high`` crudo da
    "Unexpected reasoning effort high", y ``xhigh`` inyecta la instruccion de
    pensar a fondo. Sin esta traduccion, cada peticion con esfuerzo alto a esos
    tres modelos fallaria."""
    payload = {"reasoning_effort": "high"}
    _apply_reasoning_effort(_estado(), payload)
    assert payload["chat_template_kwargs"]["reasoning_effort"] == "xhigh"


def test_low_y_medium_no_se_traducen():
    # Son vocabulario comun entre OpenAI y las plantillas de Qwen 3.5.
    for nivel in ("low", "medium"):
        payload = {"reasoning_effort": nivel}
        _apply_reasoning_effort(_estado(), payload)
        assert payload["chat_template_kwargs"]["reasoning_effort"] == nivel


def test_off_no_manda_nivel_a_la_plantilla():
    # Con enable_thinking=false la plantilla ni valida el nivel; mandarlo solo
    # podria hacerla fallar.
    payload = {"reasoning_effort": "off"}
    _apply_reasoning_effort(_estado(), payload)
    assert "reasoning_effort" not in payload["chat_template_kwargs"]


def test_arriba_va_el_nivel_traducido_no_el_crudo():
    """``reasoning_effort`` de primer nivel GANA al de ``chat_template_kwargs``.

    Este test afirmaba lo contrario —que arriba iba el valor crudo "high"
    porque solo lo leia vLLM— y por eso la suite estaba en verde mientras
    produccion devolvia 500 a toda peticion con ``reasoning_effort: "high"``
    contra los tres modelos Qwen 3.5 del parque. Comprobado en vivo contra el
    worker de qwen3.8:27b: ``{top: "high", kwargs: "xhigh"}`` da el mismo
    "Unexpected reasoning effort high" que ``high`` a secas, y solo responde
    200 cuando arriba va ya traducido.
    """
    payload = {"reasoning_effort": "high"}
    _apply_reasoning_effort(_estado(), payload)
    assert payload["reasoning_effort"] == "xhigh"
    assert payload["chat_template_kwargs"]["reasoning_effort"] == "xhigh"


def test_los_dos_sitios_coinciden_siempre(monkeypatch):
    """Mandar niveles distintos arriba y dentro es una incoherencia que la
    plantilla resuelve a su manera; nunca puede pasar."""
    monkeypatch.setattr(
        settings, "reasoning_effort_template_aliases", {"high": "xhigh", "medium": "mid"}
    )
    for nivel in ("low", "medium", "high"):
        payload = {"reasoning_effort": nivel}
        _apply_reasoning_effort(_estado(), payload)
        assert payload["reasoning_effort"] == payload["chat_template_kwargs"]["reasoning_effort"]


def test_los_alias_son_configurables(monkeypatch):
    monkeypatch.setattr(settings, "reasoning_effort_template_aliases", {"low": "minimal"})
    payload = {"reasoning_effort": "low"}
    _apply_reasoning_effort(_estado(), payload)
    assert payload["chat_template_kwargs"]["reasoning_effort"] == "minimal"


# ── vLLM: mismas reglas que llama.cpp ────────────────────────────────────────
# vLLM pasa ``chat_template_kwargs`` tal cual a la plantilla, asi que le aplican
# las mismas restricciones. La rama de vLLM enviaba ``reasoning_effort="off"``
# a pelo y sin ``enable_thinking``; comprobado renderizando la plantilla oficial
# de Qwen3.8-27B, tanto "off" como "high" dan "Unexpected reasoning effort".


def _estado_vllm():
    return SimpleNamespace(
        backend_type="vllm", extra_config={}, capabilities=SimpleNamespace(reasoning=True)
    )


def _kwargs_vllm(payload):
    return payload.get("extra_body", {}).get("chat_template_kwargs", {})


def test_vllm_apaga_el_pensamiento_con_enable_thinking():
    """Sin esta clave la plantilla entra a validar el nivel y "off" explota."""
    payload = {"reasoning_effort": "off"}
    _apply_reasoning_effort(_estado_vllm(), payload)
    kwargs = _kwargs_vllm(payload)
    assert kwargs["enable_thinking"] is False
    assert "reasoning_effort" not in kwargs


def test_vllm_traduce_high_a_xhigh():
    payload = {"reasoning_effort": "high"}
    _apply_reasoning_effort(_estado_vllm(), payload)
    assert _kwargs_vllm(payload)["reasoning_effort"] == "xhigh"


@pytest.mark.parametrize("nivel", ["low", "medium"])
def test_vllm_enciende_el_pensamiento_con_su_nivel(nivel):
    payload = {"reasoning_effort": nivel}
    _apply_reasoning_effort(_estado_vllm(), payload)
    kwargs = _kwargs_vllm(payload)
    assert kwargs["enable_thinking"] is True
    assert kwargs["reasoning_effort"] == nivel


def test_vllm_no_recibe_el_tope_duro_de_llama_cpp():
    """``reasoning_budget_tokens`` es nativo de llama-server."""
    payload = {"reasoning_effort": "low"}
    _apply_reasoning_effort(_estado_vllm(), payload)
    assert "reasoning_budget_tokens" not in payload


def test_vllm_respeta_los_kwargs_que_ya_trae_el_cliente():
    payload = {
        "reasoning_effort": "high",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False, "otra": 1}},
    }
    _apply_reasoning_effort(_estado_vllm(), payload)
    kwargs = _kwargs_vllm(payload)
    assert kwargs["enable_thinking"] is False  # el del cliente manda
    assert kwargs["otra"] == 1


# ── El presupuesto no puede comerse la respuesta ──────────────
#
# Detectado por el benchmark del 2026-11-08: `ravenx` con
# `reasoning_effort=medium` devolvia "JSON inutilizable" porque el nivel
# concedia 16.384 tokens de pensamiento sobre una salida de 4.096. El
# pensamiento se comia la respuesta entera y el cliente recibia `content`
# vacio con finish_reason "length". El informe lo describio como
# "razonamiento infinito"; no era infinito, era presupuestado por encima de
# lo que cabia.


def _presupuesto(nivel: str, max_tokens: int | None) -> int | None:
    payload: dict = {"reasoning_effort": nivel}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    _apply_reasoning_effort(_estado(), payload)
    return payload.get("reasoning_budget_tokens")


@pytest.mark.parametrize(
    ("nivel", "salida", "esperado"),
    [
        # El caso exacto del informe: medium sobre 4.096 de salida.
        ("medium", 4096, 3072),
        # Con salida holgada el nivel manda sin recortes.
        ("medium", 65536, 16384),
        ("low", 65536, 2048),
        # Salidas pequeñas: siempre queda sitio para responder.
        ("low", 512, 256),
        ("medium", 512, 256),
        # ``high`` no declara presupuesto, pero con un tope conocido se acota:
        # sin esto el pensamiento podia agotar la salida entera.
        ("high", 4096, 3072),
        ("high", 65536, 49152),
    ],
)
def test_el_presupuesto_deja_sitio_para_la_respuesta(nivel, salida, esperado):
    assert _presupuesto(nivel, salida) == esperado


@pytest.mark.parametrize(("nivel", "salida"), [("low", 512), ("medium", 4096), ("high", 4096)])
def test_nunca_se_presupuesta_mas_pensamiento_que_salida(nivel, salida):
    presupuesto = _presupuesto(nivel, salida)
    assert presupuesto is not None
    assert presupuesto < salida, "el pensamiento no puede ocupar toda la salida"


def test_sin_tope_de_salida_el_nivel_manda_sin_cambios():
    # Sin max_tokens no hay respuesta que proteger; manda el nivel.
    assert _presupuesto("low", None) == 2048
    assert _presupuesto("medium", None) == 16384
    assert _presupuesto("high", None) is None


def test_el_tope_de_ollama_tambien_cuenta():
    # En la via nativa de Ollama el limite viaja en options.num_predict.
    payload = {"reasoning_effort": "medium", "options": {"num_predict": 4096}}
    _apply_reasoning_effort(_estado(), payload)
    assert payload["reasoning_budget_tokens"] == 3072


def test_un_presupuesto_explicito_del_cliente_sigue_mandando():
    payload = {"reasoning_effort": "medium", "max_tokens": 4096, "reasoning_budget_tokens": 99}
    _apply_reasoning_effort(_estado(), payload)
    assert payload["reasoning_budget_tokens"] == 99
