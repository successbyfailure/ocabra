"""Las ``content parts`` de solo texto se aplanan para la familia llama.cpp.

Varias plantillas de chat asumen que ``message.content`` es una cadena y llaman
a ``.startswith()`` sobre ella. La que Ollama empaqueta para qwen3 es una
revision anterior a la oficial de Qwen, que añadio guardas
``message.content is string`` precisamente por esto: renderizando las dos con
``content`` como lista mas ``tools``, la de Ollama muere con
"'list object' has no attribute 'startswith'" y la oficial funciona. Afecta a
qwen3:32b y qwen3:8b, que comparten plantilla (4116 caracteres las dos).

El prompt resultante es identico: se comprobo contra el ``/apply-template`` del
propio llama-server que une las partes de texto **sin separador** y que una
sola parte equivale byte a byte a la cadena llana.
"""

from types import SimpleNamespace

import pytest

from ocabra.api.openai._deps import to_backend_body


def _estado(backend="llamacpp_mks"):
    return SimpleNamespace(
        backend_type=backend,
        backend_model_id="qwen3-32b",
        extra_config={},
        capabilities=SimpleNamespace(reasoning=False),
    )


def _cuerpo(content):
    return {"model": "qwen3:32b", "messages": [{"role": "user", "content": content}]}


def test_una_sola_parte_de_texto_se_vuelve_cadena():
    payload = to_backend_body(_estado(), _cuerpo([{"type": "text", "text": "hola"}]))
    assert payload["messages"][0]["content"] == "hola"


def test_varias_partes_se_unen_sin_separador():
    # Es como las une llama-server; cualquier separador cambiaria el prompt.
    payload = to_backend_body(
        _estado(), _cuerpo([{"type": "text", "text": "uno"}, {"type": "text", "text": "dos"}])
    )
    assert payload["messages"][0]["content"] == "unodos"


def test_una_cadena_no_se_toca():
    payload = to_backend_body(_estado(), _cuerpo("ya era una cadena"))
    assert payload["messages"][0]["content"] == "ya era una cadena"


@pytest.mark.parametrize(
    "parte",
    [
        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}},
        {"type": "video_url", "video_url": {"url": "http://x/y.mp4"}},
    ],
)
def test_una_lista_con_algo_que_no_es_texto_se_deja_intacta(parte):
    """La ruta multimodal necesita la estructura: aplanarla romperia la vision."""
    content = [{"type": "text", "text": "mira"}, parte]
    payload = to_backend_body(_estado(), _cuerpo(content))
    assert payload["messages"][0]["content"] == content


def test_una_lista_vacia_se_deja_intacta():
    payload = to_backend_body(_estado(), _cuerpo([]))
    assert payload["messages"][0]["content"] == []


def test_una_parte_sin_texto_valido_se_deja_intacta():
    content = [{"type": "text"}]  # sin clave "text"
    payload = to_backend_body(_estado(), _cuerpo(content))
    assert payload["messages"][0]["content"] == content


@pytest.mark.parametrize("backend", ["vllm", "ollama"])
def test_otros_backends_no_se_aplanan(backend):
    """vLLM y Ollama parsean las content parts ellos mismos."""
    content = [{"type": "text", "text": "hola"}]
    payload = to_backend_body(_estado(backend), _cuerpo(content))
    assert payload["messages"][0]["content"] == content


def test_no_muta_el_cuerpo_de_entrada():
    content = [{"type": "text", "text": "hola"}]
    cuerpo = _cuerpo(content)
    to_backend_body(_estado(), cuerpo)
    assert cuerpo["messages"][0]["content"] == content


def test_se_conservan_los_demas_campos_del_mensaje():
    cuerpo = {
        "model": "qwen3:32b",
        "messages": [{"role": "user", "name": "javi", "content": [{"type": "text", "text": "x"}]}],
    }
    payload = to_backend_body(_estado(), cuerpo)
    assert payload["messages"][0] == {"role": "user", "name": "javi", "content": "x"}
