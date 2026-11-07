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


def test_el_nivel_sigue_yendo_arriba_para_quien_lo_lea_de_ahi():
    payload = {"reasoning_effort": "high"}
    _apply_reasoning_effort(_estado(), payload)
    # vLLM lo expone como parametro propio, asi que no se quita de arriba.
    assert payload["reasoning_effort"] == "high"


def test_los_alias_son_configurables(monkeypatch):
    monkeypatch.setattr(settings, "reasoning_effort_template_aliases", {"low": "minimal"})
    payload = {"reasoning_effort": "low"}
    _apply_reasoning_effort(_estado(), payload)
    assert payload["chat_template_kwargs"]["reasoning_effort"] == "minimal"
