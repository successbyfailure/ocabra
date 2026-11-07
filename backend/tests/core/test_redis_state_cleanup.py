"""Al eliminar un modelo se borra tambien su estado en Redis.

``delete_model`` quitaba la fila de ``model_configs`` y el estado en memoria,
pero nunca la clave ``model:state:<id>``, y esas claves no tienen caducidad. Asi
que cada modelo eliminado o renombrado dejaba una huerfana para siempre: el
2026-11-07 habia 94 claves de 154 sin ningun modelo detras, casi todas del
renombrado de backend ``llama_cpp`` -> ``llamacpp_mks``.
"""

from unittest.mock import AsyncMock

import pytest

from ocabra import redis_client


class _RedisFalso:
    """Lo minimo de la interfaz de redis.asyncio que usan los helpers."""

    def __init__(self, claves):
        self.claves = set(claves)
        self.borradas = []

    async def delete(self, clave):
        self.borradas.append(clave)
        if clave in self.claves:
            self.claves.discard(clave)
            return 1
        return 0

    async def scan_iter(self, match=None, count=None):
        prefijo = match.removesuffix("*") if match else ""
        for c in sorted(self.claves):
            if c.startswith(prefijo):
                yield c


@pytest.fixture
def fake_redis(monkeypatch):
    r = _RedisFalso(
        [
            "model:state:llama_cpp/gemma4-26b",
            "model:state:llama_cpp/gemma4-26b::a1b2c3",
            "model:state:llama_cpp/gemma4-26b::d4e5f6",
            # de otro modelo: no debe tocarse
            "model:state:llama_cpp/gemma4-12b",
            # mismo prefijo textual pero otro modelo: tampoco
            "model:state:llama_cpp/gemma4-26b-extra",
        ]
    )
    monkeypatch.setattr(redis_client, "get_redis", lambda: r)
    return r


async def test_borra_la_clave_canonica_y_las_derivadas(fake_redis):
    n = await redis_client.delete_keys_with_prefix("model:state:llama_cpp/gemma4-26b")
    assert n == 3
    assert "model:state:llama_cpp/gemma4-26b" not in fake_redis.claves
    assert not any(c.endswith("::a1b2c3") for c in fake_redis.claves)


async def test_no_toca_otros_modelos(fake_redis):
    await redis_client.delete_keys_with_prefix("model:state:llama_cpp/gemma4-26b")
    assert "model:state:llama_cpp/gemma4-12b" in fake_redis.claves


async def test_no_confunde_un_id_que_empieza_igual(fake_redis):
    """``-extra`` no es un worker derivado: esos llevan ``::<hash>``."""
    await redis_client.delete_keys_with_prefix("model:state:llama_cpp/gemma4-26b")
    assert "model:state:llama_cpp/gemma4-26b-extra" in fake_redis.claves


async def test_una_clave_inexistente_no_falla(fake_redis):
    assert await redis_client.delete_keys_with_prefix("model:state:no/existe") == 0


async def test_delete_key_borra_una_sola(fake_redis):
    await redis_client.delete_key("model:state:llama_cpp/gemma4-12b")
    assert "model:state:llama_cpp/gemma4-12b" not in fake_redis.claves


async def test_delete_model_limpia_redis(monkeypatch):
    """El gestor tiene que invocar la limpieza al eliminar un modelo."""
    from ocabra.core import model_manager as mm_mod

    limpiador = AsyncMock(return_value=1)
    monkeypatch.setattr(mm_mod, "delete_keys_with_prefix", limpiador)

    mm = mm_mod.ModelManager(worker_pool=object())

    class _Sesion:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, *a, **k):
            return None

        async def commit(self):
            return None

    monkeypatch.setattr(mm_mod.database, "AsyncSessionLocal", lambda: _Sesion())
    await mm.delete_model("llamacpp_mks/se-va")

    limpiador.assert_awaited_once_with("model:state:llamacpp_mks/se-va")
