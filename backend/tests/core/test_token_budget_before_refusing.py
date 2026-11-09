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
    # Dos pasos: primero se le pide al worker que aplique su plantilla de chat
    # y luego se cuenta lo que devuelve. Aqui el doble no implementa
    # /apply-template (no devuelve "prompt"), asi que se cae al texto
    # concatenado y se suma el margen estructural.
    assert llamadas == [
        "http://127.0.0.1:18007/apply-template",
        "http://127.0.0.1:18007/tokenize",
    ]


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


# ── Los modelos SIN presets ──────────────────────────────────
#
# Cuatro modelos del parque no tienen presets: gemma4:26b, gemma4:12b,
# mistral:7b y ravenx. `plan_profile_runtime` sale antes por una via corta, y
# en la primera version del arreglo esa via se quedo sin afinar: el heuristico
# viajaba crudo hasta `fairness_kv_budget`, que lo compara con el pool KV del
# worker y rechaza con 400. Reproducido contra gemma4:26b el 2026-11-07: un
# prompt de 230.011 tokens reales se estimaba en 705.372 y la puerta devolvia
# "request needs 705372 KV tokens but the pool holds 262144" — 3,07x.


@pytest.mark.asyncio
async def test_sin_presets_tambien_se_afina_con_el_conteo_exacto(monkeypatch):
    from ocabra.core import runtime_coordinator as rc

    async def _exacto(mm, perfil, body):  # noqa: ARG001
        return 230_011

    monkeypatch.setattr(rc, "exact_prompt_tokens", _exacto)
    afinado, _medido = await rc.refine_required_tokens(
        _MM([]),
        _Perfil(),
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 8},
        required_tokens=705_372,
        capacity=262_144,
        default_output_reservation=None,
    )
    # 230.011 medidos + 8 de salida: cabe de sobra en el pool de 262.144.
    assert afinado == 230_019
    assert afinado < 262_144


@pytest.mark.asyncio
async def test_sin_conteo_exacto_se_intenta_en_vez_de_rechazar(monkeypatch):
    from ocabra.core import runtime_coordinator as rc

    async def _sin_medir(mm, perfil, body):  # noqa: ARG001
        return None

    monkeypatch.setattr(rc, "exact_prompt_tokens", _sin_medir)
    afinado, _medido = await rc.refine_required_tokens(
        _MM([]),
        _Perfil(),
        {"messages": []},
        required_tokens=705_372,
        capacity=262_144,
        default_output_reservation=None,
    )
    # Justo la capacidad: no rechaza, y el motor decidira de verdad.
    assert afinado == 262_144


@pytest.mark.asyncio
async def test_un_rechazo_medido_sigue_siendo_un_rechazo(monkeypatch):
    """Lo que se mide por encima de la capacidad SI se rechaza: es el caso de
    los cuatro 400 legitimos de la ronda del benchmark (46.860 medidos contra
    ventanas de 32.768 y 40.960)."""
    from ocabra.core import runtime_coordinator as rc

    async def _exacto(mm, perfil, body):  # noqa: ARG001
        return 42_764

    monkeypatch.setattr(rc, "exact_prompt_tokens", _exacto)
    afinado, _medido = await rc.refine_required_tokens(
        _MM([]),
        _Perfil(),
        {"messages": []},
        required_tokens=56_061,
        capacity=40_960,
        default_output_reservation=4096,
    )
    assert afinado == 46_860
    assert afinado > 40_960  # aguas abajo esto lanza ContextCapacityError


@pytest.mark.asyncio
async def test_lo_que_ya_cabe_no_se_toca(monkeypatch):
    """El camino rapido no paga ninguna llamada al tokenizador."""
    from ocabra.core import runtime_coordinator as rc

    llamado = {"n": 0}

    async def _contador(mm, perfil, body):  # noqa: ARG001
        llamado["n"] += 1
        return 1

    monkeypatch.setattr(rc, "exact_prompt_tokens", _contador)
    afinado, _medido = await rc.refine_required_tokens(
        _MM([]),
        _Perfil(),
        {"messages": []},
        required_tokens=5_000,
        capacity=32_768,
        default_output_reservation=None,
    )
    assert afinado == 5_000
    assert llamado["n"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("capacidad", [0, -1])
async def test_sin_capacidad_conocida_no_hay_nada_que_afinar(capacidad):
    from ocabra.core import runtime_coordinator as rc

    afinado, _medido = await rc.refine_required_tokens(
        _MM([]),
        _Perfil(),
        {"messages": []},
        required_tokens=99_999,
        capacity=capacidad,
        default_output_reservation=None,
    )
    assert afinado == 99_999


# ── Un preset pedido a mano decide el rechazo ─────────────────
#
# El afinado comparaba contra la ventana MAYOR, pero cuando el cliente pide un
# preset concreto el rechazo se compara con ESE preset. Reproducido 2026-11-08
# contra qwen3:32b con `normal` (30.720): un prompt de ~19.000 tokens reales se
# estimaba en 34.311, que es menor que la ventana mayor (40.960) y por tanto no
# se afinaba, pero mayor que el preset pedido -> 400 con el worker cargado y
# sitio de sobra. Y en el caso sin conteo exacto era peor: el recorte a 40.960
# FABRICABA un fallo que con el numero real no existia.


def _plan_args(presets, body, estados=None):
    from types import SimpleNamespace

    perfil = SimpleNamespace(
        profile_id="qwen3:32b",
        base_model_id="llamacpp_mks/qwen3:32b",
        runtime_policy={},
        load_overrides={},
        runtime_presets=presets,
    )
    return perfil, body, estados or []


@pytest.mark.asyncio
async def test_se_afina_contra_el_preset_pedido_no_contra_el_mayor(monkeypatch):
    from ocabra.core import runtime_coordinator as rc

    medidos = []

    async def _exacto(mm, perfil, body):  # noqa: ARG001
        medidos.append(True)
        return 19_000

    monkeypatch.setattr(rc, "exact_prompt_tokens", _exacto)
    # 34.311 cabe en la ventana mayor (40.960) pero no en el preset pedido
    # (30.720): antes no se afinaba y el rechazo salia con el estimado.
    afinado, _medido = await rc.refine_required_tokens(
        _MM([]),
        _Perfil(),
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 1024},
        required_tokens=34_311,
        capacity=30_720,
        default_output_reservation=None,
    )
    assert medidos, "con la capacidad del preset pedido tiene que medir"
    assert afinado == 20_024  # 19.000 medidos + 1.024 de salida
    assert afinado < 30_720, "con el numero real cabe en el preset pedido"


@pytest.mark.asyncio
async def test_sin_conteo_exacto_el_recorte_no_pasa_de_la_capacidad_decisiva(monkeypatch):
    """El recorte devuelve la capacidad que decide, nunca una mayor.

    Si devolviese la ventana mayor (40.960) contra un preset de 30.720, el
    propio recorte provocaria el 400 que intenta evitar.
    """
    from ocabra.core import runtime_coordinator as rc

    async def _sin_medir(mm, perfil, body):  # noqa: ARG001
        return None

    monkeypatch.setattr(rc, "exact_prompt_tokens", _sin_medir)
    afinado, _medido = await rc.refine_required_tokens(
        _MM([]),
        _Perfil(),
        {"messages": []},
        required_tokens=60_021,
        capacity=30_720,
        default_output_reservation=None,
    )
    assert afinado == 30_720


# --- La reserva de KV no es el presupuesto de ventana ---------------------
# Sin conteo exacto, el presupuesto se infla a proposito hasta la capacidad
# entera: asi se elige la ventana mayor y no se rechaza por una estimacion.
# Reservar ESO del pool compartido deja a los demas slots sin sitio hasta que
# la peticion termine. Visto el 2026-11-09 en flint:40b: la tarjeta marcaba
# 256k/256k de KV con una sola peticion dentro y sus cuatro slots servian de
# uno en uno.


def test_medido_se_reserva_lo_medido():
    from ocabra.core import runtime_coordinator as rc

    assert rc._reserva_kv(102_200, True, 262_144, parallelism=4) == 102_200


def test_sin_medir_no_se_reserva_el_pool_entero():
    from ocabra.core import runtime_coordinator as rc

    reserva = rc._reserva_kv(262_144, False, 262_144, parallelism=4)
    assert reserva < 262_144
    assert reserva == 65_536


def test_sin_medir_y_sin_paralelismo_se_reserva_la_ventana():
    from ocabra.core import runtime_coordinator as rc

    """Un worker de un solo slot no tiene con quien repartir."""
    assert rc._reserva_kv(262_144, False, 262_144, parallelism=1) == 262_144
    assert rc._reserva_kv(262_144, False, 262_144, parallelism=None) == 262_144


def test_sin_capacidad_conocida_se_deja_lo_que_haya():
    from ocabra.core import runtime_coordinator as rc

    assert rc._reserva_kv(1234, False, 0, parallelism=4) == 1234


def test_la_reserva_nunca_es_negativa():
    from ocabra.core import runtime_coordinator as rc

    assert rc._reserva_kv(-5, True, 262_144, parallelism=4) == 0


@pytest.mark.asyncio
async def test_con_la_plantilla_aplicada_el_conteo_es_exacto(monkeypatch):
    """Contar el prompt renderizado da el numero de verdad, no una aproximacion.

    ``/apply-template`` devuelve exactamente lo que vera el modelo —marcas de
    rol, formato de las llamadas a herramientas, esquemas de ``tools``—, asi
    que no hay que sumarle ningun margen estructural. Medido el 2026-11-09
    contra flint:40b: la plantilla aplicada da 7.833 tokens y el motor cuenta
    7.833; concatenar el texto a mano daba 8.490.
    """
    from ocabra.core.runtime_coordinator import exact_prompt_tokens

    cuerpos = []

    class _Resp:
        status_code = 200

        def __init__(self, datos):
            self._datos = datos

        def raise_for_status(self):
            pass

        def json(self):
            return self._datos

    class _Cliente:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            cuerpos.append((url, json))
            if url.endswith("/apply-template"):
                return _Resp({"prompt": "<|user|>hola<|assistant|>"})
            return _Resp({"tokens": list(range(900))})

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Cliente)
    cuerpo = {"messages": [{"role": "user", "content": "hola"}], "tools": [{"a": 1}]}
    total = await exact_prompt_tokens(_MM([_Estado("m")]), _Perfil(base_model_id="m"), cuerpo)

    # 900 pelados, sin margen: la plantilla ya trae el andamiaje dentro.
    assert total == 900
    # Y a la plantilla se le pasan los mensajes y las tools, que es donde
    # viven los esquemas grandes.
    assert cuerpos[0][1] == {"messages": cuerpo["messages"], "tools": cuerpo["tools"]}
    assert cuerpos[1][1] == {"content": "<|user|>hola<|assistant|>"}
