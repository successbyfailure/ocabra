"""Quien pregunta la version tiene que recibirla.

``/version`` no era una ruta de la API, asi que el proxy la dejaba caer al SPA
del frontend y lo que volvia era el 404 de su nginx. Y ``/api/version`` —que si
es parte de la API de Ollama, la que usa el propio cliente ``ollama`` para
decidir que soporta el servidor antes de mandar nada— no existia.

Reportado el 2026-11-08 por un arnes de benchmark que no podia anotar con que
version se midio cada cosa. Las dos rutas van sin autenticacion, igual que
``/health`` y que Ollama: es como un cliente identifica al servidor.

Ojo: esto cubre el enrutado de FastAPI, no el de Caddy. ``/version`` necesita
ademas su propia regla en ``caddy/Caddyfile``.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api.health import router as health_router
from ocabra.api.ollama import router as ollama_router
from ocabra.config import settings


@pytest.fixture
def cliente() -> TestClient:
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(ollama_router)
    return TestClient(app)


@pytest.mark.parametrize("ruta", ["/version", "/api/version"])
def test_la_version_se_sirve_en_las_dos_rutas(cliente, ruta):
    respuesta = cliente.get(ruta)
    assert respuesta.status_code == 200
    assert respuesta.json() == {"version": settings.app_version}


@pytest.mark.parametrize("ruta", ["/version", "/api/version"])
def test_no_hace_falta_clave_para_preguntar_la_version(cliente, ruta):
    """Sin cabecera de autorizacion ninguna."""
    assert cliente.get(ruta).status_code == 200


def test_health_sigue_trayendo_la_version(cliente):
    """Era la unica via antes y hay clientes que la usan."""
    cuerpo = cliente.get("/health").json()
    assert cuerpo == {"status": "ok", "version": settings.app_version}


def test_las_dos_rutas_dicen_lo_mismo(cliente):
    assert cliente.get("/version").json() == cliente.get("/api/version").json()
