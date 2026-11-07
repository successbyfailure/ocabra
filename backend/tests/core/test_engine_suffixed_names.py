"""Los nombres que llevan el motor dentro (``gemma4:26b-ollama``) no son públicos.

``_create_pinned_variant`` crea un perfil oculto por cada despliegue que no es
el por defecto, y lo nombra con el motor como sufijo para garantizar que todo
despliegue es alcanzable. Ese nombre ata al cliente a un motor que el servidor
puede cambiar, así que no se puede pedir: se pide el modelo por su nombre, o el
despliegue por su id (``ollama/gemma4:26b``).

Los perfiles con nombre propio (``whisper-cpp-large-v3-turbo-cpu``, destino de
``recomendado-stt``) sí son públicos y no deben caer en la regla.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from ocabra.core.profile_registry import ProfileRegistry
from ocabra.db.model_config import ModelProfile


def _perfil(profile_id: str, base_model_id: str, backend: str | None) -> ModelProfile:
    return ModelProfile(
        profile_id=profile_id,
        base_model_id=base_model_id,
        backend=backend,
        category="llm",
        enabled=True,
        is_default=True,
    )


@pytest.fixture
def registro() -> ProfileRegistry:
    registro = ProfileRegistry()
    registro._profiles = {
        p.profile_id: p
        for p in [
            # Perfiles públicos: el nombre del modelo, sin motor.
            _perfil("gemma4:26b", "llamacpp_mks/gemma4:26b", None),
            _perfil("qwen3-coder:30b", "llamacpp_mks/qwen3-coder:30b", None),
            _perfil("mistral:7b", "llamacpp_mks/mistral:7b", None),
            # Variantes auto-derivadas: <model_key>-<backend>.
            _perfil("gemma4:26b-ollama", "ollama/gemma4:26b", "ollama"),
            _perfil("qwen3-coder:30b-ollama", "ollama/qwen3-coder:30b", "ollama"),
            _perfil("mistral:7b-llamacpp-mks", "llamacpp_mks/mistral:7b", "llamacpp_mks"),
            # Variante con nombre propio: pública, no se cierra.
            _perfil(
                "whisper-cpp-large-v3-turbo-cpu",
                "whisper_cpp/whisper-large-v3-turbo",
                "whisper_cpp",
            ),
            # Un modelo que solo existe en Ollama: su perfil público lleva
            # backend a None y no tiene la forma auto-derivada.
            _perfil("nemotron3:33b", "ollama/nemotron3:33b", None),
        ]
    }
    registro._aliases = {}
    return registro


# ── El detector ──────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("nombre", "esperado"),
    [
        ("gemma4:26b-ollama", ("gemma4:26b", "ollama/gemma4:26b")),
        ("qwen3-coder:30b-ollama", ("qwen3-coder:30b", "ollama/qwen3-coder:30b")),
        ("mistral:7b-llamacpp-mks", ("mistral:7b", "llamacpp_mks/mistral:7b")),
    ],
)
async def test_detecta_la_forma_auto_derivada(registro, nombre, esperado):
    assert await registro.engine_suffixed_name(nombre) == esperado


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "nombre",
    [
        "gemma4:26b",  # nombre público
        "qwen3-coder:30b",
        "nemotron3:33b",  # publico, aunque su despliegue sea de Ollama
        "whisper-cpp-large-v3-turbo-cpu",  # variante con nombre propio
        "no-existe",
        "",
    ],
)
async def test_no_toca_lo_que_si_es_publico(registro, nombre):
    assert await registro.engine_suffixed_name(nombre) is None


@pytest.mark.asyncio
async def test_el_guion_bajo_del_backend_se_normaliza_a_guion(registro):
    # El backend es ``llamacpp_mks`` pero el perfil se llama
    # ``-llamacpp-mks``: si la comparacion no normalizase, no casaria.
    assert await registro.engine_suffixed_name("mistral:7b-llamacpp-mks") is not None
    assert await registro.engine_suffixed_name("mistral:7b-llamacpp_mks") is None


@pytest.mark.asyncio
async def test_un_perfil_cuyo_base_no_empieza_por_su_backend_no_casa():
    registro = ProfileRegistry()
    # Incoherencia defensiva: backend "ollama" pero el despliegue es de otro.
    registro._profiles = {
        "x:1b-ollama": _perfil("x:1b-ollama", "llamacpp_mks/x:1b", "ollama"),
    }
    registro._aliases = {}
    assert await registro.engine_suffixed_name("x:1b-ollama") is None


# ── La puerta de los endpoints OpenAI ────────────────────────


@pytest.mark.asyncio
async def test_resolve_profile_rechaza_el_nombre_con_motor(registro):
    from ocabra.api.openai._deps import resolve_profile

    with pytest.raises(HTTPException) as exc:
        await resolve_profile("gemma4:26b-ollama", object(), registro, request_body={})
    assert exc.value.status_code == 404
    mensaje = str(exc.value.detail)
    # El error tiene que llevar las dos salidas, no solo "no existe".
    assert "gemma4:26b" in mensaje
    assert "ollama/gemma4:26b" in mensaje


@pytest.mark.asyncio
async def test_resolve_profile_no_toca_los_nombres_propios(registro):
    from ocabra.api.openai._deps import resolve_profile

    # No llega a resolver (el ModelManager es un stub), pero el fallo NO puede
    # ser el 404 de la puerta: tiene que pasar de largo.
    with pytest.raises(Exception) as exc:  # noqa: B017 — cualquier otro fallo sirve
        await resolve_profile("whisper-cpp-large-v3-turbo-cpu", object(), registro, request_body={})
    assert "if you need that specific engine" not in str(getattr(exc.value, "detail", ""))


# ── La puerta de los endpoints Ollama ────────────────────────


@pytest.mark.asyncio
async def test_resolve_ollama_profile_rechaza_el_nombre_con_motor(registro):
    from ocabra.api.ollama._mapper import resolve_ollama_profile

    with pytest.raises(HTTPException) as exc:
        await resolve_ollama_profile(object(), registro, "qwen3-coder:30b-ollama")
    assert exc.value.status_code == 404
    mensaje = str(exc.value.detail)
    assert "qwen3-coder:30b" in mensaje
    assert "ollama/qwen3-coder:30b" in mensaje


@pytest.mark.asyncio
async def test_en_ollama_el_404_no_se_lo_come_el_fallback(registro):
    """El ``except`` de ``resolve_ollama_profile`` traga el 404 de
    ``resolve_profile``; por eso la puerta va antes del ``try``."""
    from ocabra.api.ollama._mapper import resolve_ollama_profile

    with pytest.raises(HTTPException):
        await resolve_ollama_profile(object(), registro, "  gemma4:26b-ollama  ")


# ── Compatibilidad con registros que no implementan la puerta ─


@pytest.mark.asyncio
async def test_un_registro_sin_el_metodo_no_rompe_la_resolucion():
    """La puerta se consulta con ``getattr``: los dobles de test de otras
    suites no implementan ``engine_suffixed_name`` y deben seguir pasando."""
    from ocabra.api.openai._deps import resolve_profile

    class _SinPuerta:
        async def resolve_alias(self, _alias_id):
            return None

        async def get(self, _profile_id):
            return None

    with pytest.raises(HTTPException) as exc:
        await resolve_profile("lo-que-sea", _Stub(), _SinPuerta(), request_body={})
    assert exc.value.status_code == 404


class _Stub:
    async def get_state(self, _model_id):
        return None
