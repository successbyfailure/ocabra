"""/v1/videos: validación, encolado y el workflow de LTX-2.5 del worker."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocabra.api._deps_auth import UserContext, get_current_user
from ocabra.core.admission import should_track
from ocabra.core.video_jobs import normalize_size

WORKERS = Path(__file__).resolve().parents[1] / "workers"
MODULE = "ocabra.api.openai.videos"


def _worker_module():
    spec = importlib.util.spec_from_file_location(
        "comfyui_video_worker", WORKERS / "comfyui_video_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("comfyui_video_worker", module)
    spec.loader.exec_module(module)
    return module


# ── Tamaño y parámetros ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("asked", "served"),
    [((1280, 720), (1280, 704)), ((720, 1280), (704, 1280)), ((100, 100), (256, 256))],
)
def test_el_tamano_se_redondea_a_multiplos_de_64(asked, served):
    assert normalize_size(*asked) == served


def test_videos_no_pasa_por_el_registro_automatico_de_estadisticas():
    # El POST solo encola; cada trabajo registra su fila al terminar.
    assert should_track("/v1/videos") is False
    assert should_track("/v1/videos/video_abc/content") is False
    assert should_track("/v1/images/generations") is True


# ── Workflow del worker ────────────────────────────────────────────────


def test_el_workflow_rellena_prompt_tamano_duracion_y_semilla():
    worker = _worker_module()
    template = worker.WorkflowTemplate(WORKERS / "comfyui_video_workflows" / "ltx25_t2v.json")
    prompt, seed = template.render(
        {"prompt": "un taller", "width": 960, "height": 512, "seconds": 4, "seed": 7}, "ocabra/x"
    )
    p = template.params
    assert prompt[p["prompt"]]["inputs"]["value"] == "un taller"
    assert prompt[p["width"]]["inputs"]["value"] == 960
    assert prompt[p["height"]]["inputs"]["value"] == 512
    assert prompt[p["seconds"]]["inputs"]["value"] == 4
    assert prompt[p["seed"]]["inputs"]["noise_seed"] == 7 and seed == 7
    assert prompt[p["output"]]["inputs"]["filename_prefix"] == "ocabra/x"
    # La plantilla no se modifica entre peticiones.
    assert template.prompt[p["prompt"]]["inputs"]["value"] == ""


def test_el_workflow_ordena_los_muestreadores_y_no_usa_el_mejorador():
    worker = _worker_module()
    path = WORKERS / "comfyui_video_workflows" / "ltx25_t2v.json"
    template = worker.WorkflowTemplate(path)
    first, second = template.samplers
    # El refinado toma el latente de la primera pasada, nunca al revés.
    assert template._depends_on_sampler(second, template.samplers) == 1
    assert template._depends_on_sampler(first, template.samplers) == 0
    # La rama del mejorador necesita un modelo que no se descarga.
    classes = {node["class_type"] for node in json.loads(path.read_text())["prompt"].values()}
    assert "TextGenerateLTX2Prompt" not in classes


def test_el_backend_solo_acepta_nombres_de_workflow_del_directorio():
    from ocabra.backends.comfyui_video_backend import ComfyUIVideoBackend

    assert ComfyUIVideoBackend.workflow_path("ltx25_t2v").name == "ltx25_t2v.json"
    for bad in ("../secret", "/etc/passwd", ".hidden"):
        with pytest.raises(ValueError):
            ComfyUIVideoBackend.workflow_path(bad)


# ── POST /v1/videos ────────────────────────────────────────────────────


def _app(*, capable=True, profile=True):
    from ocabra.api.openai import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="7f1c2a3e-1111-4222-8333-944455556666", username="ana", role="user",
        accessible_model_ids={"ltx-2.5", "comfyui_video/ltx-2.5"},
    )  # fmt: skip
    registry = MagicMock()
    registry.get = AsyncMock(
        return_value=SimpleNamespace(
            profile_id="ltx-2.5", base_model_id="comfyui_video/ltx-2.5", enabled=True
        )
        if profile
        else None
    )
    manager = MagicMock()
    manager.get_state = AsyncMock(
        return_value=SimpleNamespace(capabilities=SimpleNamespace(video_generation=capable))
    )
    jobs = MagicMock()
    jobs.submit = AsyncMock()
    app.state.profile_registry = registry
    app.state.model_manager = manager
    app.state.video_job_manager = jobs
    return app, jobs


class _Session:
    def __init__(self):
        self.added = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        for row in self.added:
            row.created_at = __import__("datetime").datetime.now()

    async def refresh(self, row):
        return None


def _post(app, body):
    session = _Session()
    with patch(f"{MODULE}.AsyncSessionLocal", lambda: session):
        resp = TestClient(app, raise_server_exceptions=False).post("/v1/videos", json=body)
    return resp, session


def test_encola_y_devuelve_el_objeto_video_con_el_tamano_real():
    app, jobs = _app()
    resp, session = _post(
        app, {"model": "ltx-2.5", "prompt": "un taller", "seconds": "4", "size": "1280x720"}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["object"] == "video" and data["status"] == "queued"
    assert data["id"].startswith("video_")
    assert data["size"] == "1280x704" and data["seconds"] == "4"
    pending = jobs.submit.await_args.args[0]
    assert pending.params["prompt"] == "un taller"
    # El prompt no va a la fila de la base de datos.
    assert not hasattr(session.added[0], "prompt")


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"model": "ltx-2.5"}, "prompt"),
        ({"model": "ltx-2.5", "prompt": "x", "seconds": "60"}, "seconds"),
        ({"model": "ltx-2.5", "prompt": "x", "size": "grande"}, "size"),
        ({"model": "ltx-2.5", "prompt": "x", "size": "4096x4096"}, "size"),
        ({"prompt": "x"}, "model"),
    ],
)
def test_rechaza_parametros_invalidos(body, param):
    app, jobs = _app()
    resp, _ = _post(app, body)
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["param"] == param
    jobs.submit.assert_not_awaited()


def test_un_modelo_que_no_genera_video_da_400():
    app, jobs = _app(capable=False)
    resp, _ = _post(app, {"model": "ltx-2.5", "prompt": "x"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "model_not_capable"


def test_un_modelo_inexistente_da_404():
    app, _ = _app(profile=False)
    app.state.model_manager.get_state = AsyncMock(return_value=None)
    resp, _ = _post(app, {"model": "no-existe", "prompt": "x"})
    assert resp.status_code == 404
