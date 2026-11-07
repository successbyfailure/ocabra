"""Una peticion no-streaming se cancela cuando el cliente se desconecta.

En la via de streaming Starlette ya cancela la tarea al irse el cliente, y esa
cancelacion cierra la conexion con el worker, lo que hace que el motor aborte y
libere la GPU (contrato en ``WorkerPool._raw_stream``). En la via NO de
streaming FastAPI deja correr el handler hasta el final.

Medido el 2026-11-07: un cliente se rindio a los ~900 s y oCabra siguio
generando hasta los 1500 s de su propio timeout, con el slot de admision y la
GPU ocupados diez minutos de mas para una respuesta que ya no leia nadie.
"""

import asyncio
from types import SimpleNamespace

import pytest

from ocabra.api.openai._deps import ClientGoneError, await_unless_client_gone


def _peticion(desconectado):
    """Doble de Request: ``desconectado`` es un callable que devuelve bool."""

    async def _is_disconnected():
        return desconectado()

    return SimpleNamespace(is_disconnected=_is_disconnected)


async def test_devuelve_el_resultado_si_el_cliente_sigue_ahi():
    async def trabajo():
        await asyncio.sleep(0)
        return "listo"

    assert await await_unless_client_gone(
        _peticion(lambda: False), trabajo(), poll_seconds=0.01
    ) == "listo"


async def test_propaga_la_excepcion_del_trabajo():
    async def trabajo():
        raise ValueError("algo fallo arriba")

    with pytest.raises(ValueError, match="algo fallo"):
        await await_unless_client_gone(_peticion(lambda: False), trabajo(), poll_seconds=0.01)


async def test_cancela_el_trabajo_si_el_cliente_se_va():
    cancelado = asyncio.Event()

    async def trabajo():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelado.set()
            raise
        return "no deberia llegar"

    with pytest.raises(ClientGoneError):
        await await_unless_client_gone(_peticion(lambda: True), trabajo(), poll_seconds=0.01)
    # Lo que de verdad importa: la generacion se aborta y el slot se suelta.
    assert cancelado.is_set()


async def test_un_trabajo_que_acaba_a_tiempo_gana_a_la_desconexion():
    """Si ya hay resultado no se descarta por una desconexion simultanea."""

    async def trabajo():
        return "a tiempo"

    assert await await_unless_client_gone(
        _peticion(lambda: True), trabajo(), poll_seconds=5
    ) == "a tiempo"


async def test_una_sonda_que_falla_no_rompe_la_peticion():
    """``is_disconnected`` puede fallar segun el servidor ASGI; nunca debe
    tumbar una peticion que va bien."""

    async def _revienta():
        raise RuntimeError("sin canal de recepcion")

    peticion = SimpleNamespace(is_disconnected=_revienta)

    async def trabajo():
        await asyncio.sleep(0.03)
        return "listo"

    assert await await_unless_client_gone(peticion, trabajo(), poll_seconds=0.01) == "listo"


async def test_si_cancelan_desde_fuera_tambien_cancela_el_trabajo():
    cancelado = asyncio.Event()

    async def trabajo():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelado.set()
            raise

    tarea = asyncio.ensure_future(
        await_unless_client_gone(_peticion(lambda: False), trabajo(), poll_seconds=0.01)
    )
    await asyncio.sleep(0.05)
    tarea.cancel()
    with pytest.raises(asyncio.CancelledError):
        await tarea
    assert cancelado.is_set()
