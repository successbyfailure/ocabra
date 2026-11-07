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
