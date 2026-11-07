"""El vigilante de recarga de un PIN no se rinde a los 12 minutos.

``_watch_and_reload`` tenia un plazo de ``model_load_wait_timeout_s`` (720 s):
a quien se quedase la GPU le bastaba con aguantarla 12 minutos y el PIN se
quedaba fuera hasta el siguiente reinicio del proceso. Es exactamente lo que
hace un benchmark, o cualquier uso sostenido de otro modelo grande.

Reproducido el 2026-11-07: ``gemma4:26b`` (23.600 MB de los 24.576 de la 3090)
cargó a las 21:18:38, una sola peticion a ``qwen3.8:27b`` lo desalojo a las
21:18:55, y a partir de ahi toda peticion respondia 409 — ni desalojando todo
lo que quedaba en la tarjeta se liberaban los 23.600 que pide, asi que el
preflight lo veia imposible y ni lo intentaba.

La conducta buscada: mientras el modelo siga siendo PIN y siga descargado, se
sigue comprobando si hay sitio, sin plazo. Nunca pelea por la tarjeta
(``allow_pressure_eviction=False``): solo entra si de verdad esta libre.
"""

from __future__ import annotations

import asyncio

import pytest

from ocabra.config import settings
from ocabra.core.model_manager import LoadPolicy, ModelManager, ModelStatus


class _Backend:
    async def get_vram_estimate_mb(self, model_id, extra_config=None):  # noqa: ARG002
        return 23600


class _WorkerPool:
    async def get_backend(self, backend_type):  # noqa: ARG002
        return _Backend()


class _Scheduler:
    """Niega la GPU las ``negativas`` primeras veces y luego la concede."""

    def __init__(self, negativas: int):
        self.negativas = negativas
        self.consultas = 0

    async def find_gpu_for_model(self, vram_needed, preferred_gpu=None):  # noqa: ARG002
        self.consultas += 1
        if self.consultas <= self.negativas:
            return []
        return [1]


def _manager(negativas: int, estado_modelo) -> tuple[ModelManager, list[tuple], _Scheduler]:
    mm = ModelManager(worker_pool=_WorkerPool())
    scheduler = _Scheduler(negativas)
    mm._gpu_scheduler = scheduler
    mm._states = {"llamacpp_mks/gemma4:26b": estado_modelo}
    mm._stopped = False
    cargas: list[tuple] = []

    async def _load(model_id, allow_pressure_eviction=True, **kwargs):  # noqa: ARG001
        cargas.append((model_id, allow_pressure_eviction))
        estado_modelo.status = ModelStatus.LOADED

    mm._load_model = _load
    return mm, cargas, scheduler


class _Estado:
    def __init__(self):
        self.model_id = "llamacpp_mks/gemma4:26b"
        self.backend_type = "llamacpp_mks"
        self.backend_model_id = "gemma4:26b"
        self.load_policy = LoadPolicy.PIN
        self.status = ModelStatus.UNLOADED
        self.preferred_gpu = 1
        self.extra_config = {}
        self.reload_suppressed_until = None


@pytest.fixture(autouse=True)
def _sin_esperas(monkeypatch):
    """El bucle duerme entre intentos; aqui se cuentan las vueltas, no el reloj.

    No basta con poner ``pin_reload_poll_interval_s`` a 0: el bucle aplica un
    suelo de ``max(1, ...)``, asi que 41 vueltas serian 41 segundos de test.
    Se sustituye el ``sleep`` del modulo por un cambio de contexto a secas.
    """
    import ocabra.core.model_manager as mm_mod

    async def _ceder(_segundos):
        await original(0)

    original = mm_mod.asyncio.sleep
    monkeypatch.setattr(mm_mod.asyncio, "sleep", _ceder)
    monkeypatch.setattr(settings, "pin_reload_poll_interval_s", 0)


@pytest.mark.asyncio
async def test_sigue_intentandolo_mucho_despues_del_plazo_viejo():
    # 12 minutos a 120 s por vuelta eran ~6 intentos: con 40 negativas el
    # vigilante antiguo se habria rendido hace rato.
    estado = _Estado()
    mm, cargas, scheduler = _manager(negativas=40, estado_modelo=estado)
    await asyncio.wait_for(mm._watch_and_reload(estado.model_id), timeout=10)
    assert scheduler.consultas == 41
    assert cargas == [("llamacpp_mks/gemma4:26b", False)]


@pytest.mark.asyncio
async def test_nunca_desaloja_para_entrar():
    """Si forzase la entrada desalojaria al modelo que gano la tarjeta y los
    dos se turnarian la GPU indefinidamente."""
    estado = _Estado()
    mm, cargas, _ = _manager(negativas=0, estado_modelo=estado)
    await asyncio.wait_for(mm._watch_and_reload(estado.model_id), timeout=10)
    assert cargas and cargas[0][1] is False


@pytest.mark.asyncio
async def test_deja_de_vigilar_si_el_modelo_se_carga_por_otra_via():
    estado = _Estado()
    estado.status = ModelStatus.LOADED
    mm, cargas, scheduler = _manager(negativas=0, estado_modelo=estado)
    await asyncio.wait_for(mm._watch_and_reload(estado.model_id), timeout=10)
    assert cargas == []
    assert scheduler.consultas == 0


@pytest.mark.asyncio
async def test_deja_de_vigilar_si_el_pin_se_retira():
    estado = _Estado()
    estado.load_policy = LoadPolicy.ON_DEMAND
    mm, cargas, _ = _manager(negativas=0, estado_modelo=estado)
    await asyncio.wait_for(mm._watch_and_reload(estado.model_id), timeout=10)
    assert cargas == []


@pytest.mark.asyncio
async def test_para_al_apagarse():
    estado = _Estado()
    mm, cargas, _ = _manager(negativas=10**6, estado_modelo=estado)
    mm._stopped = True
    await asyncio.wait_for(mm._watch_and_reload(estado.model_id), timeout=10)
    assert cargas == []


@pytest.mark.asyncio
async def test_no_arranca_dos_vigilantes_para_el_mismo_modelo():
    """Un unload puede notificarse por dos vias; dos bucles doblarian el
    sondeo y podrian lanzar dos cargas del mismo modelo."""
    estado = _Estado()
    mm, cargas, _ = _manager(negativas=3, estado_modelo=estado)
    primero = asyncio.create_task(mm._watch_and_reload(estado.model_id))
    await asyncio.sleep(0)
    segundo = await mm._watch_and_reload(estado.model_id)
    assert segundo is None
    await asyncio.wait_for(primero, timeout=10)
    assert len(cargas) == 1


@pytest.mark.asyncio
async def test_al_terminar_se_libera_el_hueco_del_vigilante():
    estado = _Estado()
    mm, _, _ = _manager(negativas=0, estado_modelo=estado)
    await asyncio.wait_for(mm._watch_and_reload(estado.model_id), timeout=10)
    assert estado.model_id not in mm._pin_reload_watchers


@pytest.mark.asyncio
async def test_respeta_una_suspension_pedida_por_un_servicio():
    """``ensure_vram_free`` puede pedir que no se recargue durante un rato,
    por ejemplo mientras un servicio externo usa la tarjeta."""
    from datetime import UTC, datetime, timedelta

    estado = _Estado()
    estado.reload_suppressed_until = datetime.now(UTC) + timedelta(seconds=30)
    mm, cargas, scheduler = _manager(negativas=0, estado_modelo=estado)

    # La suspension se levanta sola: el sleep del bucle esta parcheado, asi
    # que se usa un reloj falso que avanza en cada vuelta.
    vueltas = {"n": 0}
    original_now = datetime.now

    class _Reloj(datetime):
        @classmethod
        def now(cls, tz=None):
            vueltas["n"] += 1
            base = original_now(tz)
            return base + timedelta(seconds=60) if vueltas["n"] > 4 else base

    import ocabra.core.model_manager as mm_mod

    mm_mod.datetime = _Reloj
    try:
        await asyncio.wait_for(mm._watch_and_reload(estado.model_id), timeout=10)
    finally:
        mm_mod.datetime = datetime
    # Mientras estuvo suspendido no consulto la GPU ni una vez.
    assert scheduler.consultas == 1
    assert cargas == [("llamacpp_mks/gemma4:26b", False)]
