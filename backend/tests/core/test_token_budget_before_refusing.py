"""Un prompt no se rechaza por tamaño basandose en una estimacion.

Medido el 2026-11-07 contra el ``/tokenize`` del worker de qwen3.8:27b, el
heuristico de ``estimate_prompt_tokens`` va de 0,94x a 2,42x del conteo real
segun el contenido:

    changelog ES (markdown denso)   4.031 reales   chars/3 = 4.388   0,94x–1,26x
    codigo Python                   4.812 reales   chars/3 = 7.082   1,70x
    palabras largas                39.888 reales   chars/3 = 96.636  2,42x

Y ``words*1.5`` subestima en las seis muestras. Ninguna constante sirve a la
vez para prosa densa y para texto de palabras largas, asi que un rechazo no
puede decidirse con el heuristico: un prompt de ~32k tokens reales se estimaba
en ~55k y ``qwen3:32b`` (ventana de 40.960) lo rechazaba con 400 aunque cabia.

El heuristico se queda para ELEGIR ventana, donde equivocarse solo carga una
mas grande de lo necesario.
"""

from __future__ import annotations

import pytest

from ocabra.core.runtime_coordinator import (
    _prompt_text,
    estimate_prompt_tokens,
    exact_prompt_tokens,
    output_reservation,
)

# ── La reserva de salida, ahora factorizada ──────────────────


@pytest.mark.parametrize(
    ("body", "esperado"),
    [
        ({"max_tokens": 500}, 500),
        ({"max_completion_tokens": 700}, 700),
        ({"options": {"num_predict": 300}}, 300),
        ({}, 4096),  # el default que se le pasa
        ({"max_tokens": "basura"}, 4096),
        ({"max_tokens": 0}, 1),  # nunca cero: al menos un token de salida
    ],
)
def test_la_reserva_de_salida_sale_del_cuerpo(body, esperado):
    assert output_reservation(body, 4096) == esperado


def test_max_tokens_explicito_gana_al_default():
    # Es el mismo orden de precedencia que ya aplicaba el estimador completo;
    # factorizarlo no puede haberlo cambiado.
    assert output_reservation({"max_tokens": 10, "options": {"num_predict": 99}}, 4096) == 10


# ── El texto que se manda a tokenizar ────────────────────────


def test_el_texto_recoge_todos_los_mensajes():
    cuerpo = {
        "messages": [
            {"role": "system", "content": "eres util"},
            {"role": "user", "content": "hola"},
        ]
    }
    texto = _prompt_text(cuerpo)
    assert "eres util" in texto
    assert "hola" in texto


def test_el_texto_recoge_las_content_parts_de_texto():
    cuerpo = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "primera"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "text", "text": "segunda"},
                ],
            }
        ]
    }
    texto = _prompt_text(cuerpo)
    assert "primera" in texto and "segunda" in texto
    # La imagen no es texto tokenizable; su coste lo sigue poniendo el
    # heuristico via _IMAGE_TOKEN_RESERVATION.
    assert "data:" not in texto


def test_el_texto_incluye_los_tools():
    cuerpo = {"messages": [{"role": "user", "content": "x"}], "tools": [{"name": "buscar"}]}
    assert "buscar" in _prompt_text(cuerpo)


def test_el_texto_vale_para_completions_y_embeddings():
    assert "abc" in _prompt_text({"prompt": "abc"})
    assert "def" in _prompt_text({"input": "def"})
    assert "g" in _prompt_text({"input": ["g", "h"]})


# ── El conteo exacto ─────────────────────────────────────────


class _Estado:
    def __init__(self, model_id, backend_type="llamacpp_mks", cargado=True, puerto=18004):
        self.model_id = model_id
        self.backend_type = backend_type
        self.status = type("E", (), {"value": "loaded" if cargado else "unloaded"})()
        self.worker_info = type("W", (), {"port": puerto})() if puerto else None


class _MM:
    def __init__(self, estados):
        self._estados = estados

    async def list_states(self, include_derived=False):  # noqa: ARG002
        return self._estados


class _Perfil:
    def __init__(self, base_model_id="llamacpp_mks/qwen3:32b"):
        self.base_model_id = base_model_id
        self.profile_id = "qwen3:32b"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "estados",
    [
        [],  # nada cargado
        [_Estado("llamacpp_mks/qwen3:32b", cargado=False)],  # cargado=no
        [_Estado("llamacpp_mks/otro:7b")],  # otro modelo
        [_Estado("llamacpp_mks/qwen3:32b", backend_type="ollama")],  # otro backend
        [_Estado("llamacpp_mks/qwen3:32b", puerto=None)],  # sin worker
    ],
)
async def test_sin_worker_utilizable_no_hay_conteo_exacto(estados):
    # ``None`` es la senal de "no rechaces por estimacion", no un cero.
    assert await exact_prompt_tokens(_MM(estados), _Perfil(), {"messages": []}) is None


@pytest.mark.asyncio
async def test_un_perfil_sin_modelo_base_no_intenta_nada():
    assert await exact_prompt_tokens(_MM([]), _Perfil(base_model_id=None), {}) is None


@pytest.mark.asyncio
async def test_usa_el_tokenizador_de_un_derivado_del_mismo_modelo(monkeypatch):
    """Los runtimes derivados (``<base>::<hash>``, un preset distinto) comparten
    tokenizador con su base, asi que sirven igual."""
    llamadas = []

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"tokens": list(range(1234))}

    class _Cliente:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json):  # noqa: A002
            llamadas.append(url)
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Cliente)
    estados = [_Estado("llamacpp_mks/qwen3:32b::abc123", puerto=18007)]
    cuerpo = {"messages": [{"role": "user", "content": "hola"}]}
    total = await exact_prompt_tokens(_MM(estados), _Perfil(), cuerpo)
    # 1234 del tokenizador + 16 de margen estructural por mensaje.
    assert total == 1234 + 16
    assert llamadas == ["http://127.0.0.1:18007/tokenize"]


@pytest.mark.asyncio
async def test_si_el_tokenizador_falla_devuelve_none(monkeypatch):
    class _Cliente:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json):  # noqa: A002
            raise TimeoutError("el worker no contesta")

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Cliente)
    estados = [_Estado("llamacpp_mks/qwen3:32b")]
    assert await exact_prompt_tokens(_MM(estados), _Perfil(), {"messages": []}) is None


# ── Lo que motivo el cambio ──────────────────────────────────


def test_el_heuristico_sobreestima_el_caso_reportado():
    """Deja constancia del tamaño del error, para que si alguien afina el
    heuristico se vea si mejora o no."""
    # ~25.000 palabras de palabras largas: el caso del benchmark.
    texto = " ".join(["configuracion", "rendimiento", "documentacion"] * 8334)
    estimado = estimate_prompt_tokens({"messages": [{"role": "user", "content": texto}]})
    # Medido con el tokenizador de Qwen: ~1,6 tokens por palabra.
    reales_aprox = len(texto.split()) * 1.6
    assert estimado > reales_aprox * 1.5, (
        f"estimado {estimado} frente a ~{reales_aprox:.0f} reales; "
        "si esto deja de cumplirse, el heuristico ha mejorado"
    )
    # Y el exceso basta para pasarse de la ventana de qwen3:32b.
    assert estimado > 40960
