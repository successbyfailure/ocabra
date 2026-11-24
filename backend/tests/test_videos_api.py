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


def _app(*, capable=True, profile=True, backend_type="comfyui_video"):
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
        return_value=SimpleNamespace(
            capabilities=SimpleNamespace(video_generation=capable),
            backend_type=backend_type,
            backend_model_id="video",
            model_id="comfyui_video/video",
            extra_config={},
        )
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


def test_un_modelo_de_video_sin_cargar_se_acepta():
    # Sin cargar, sus capacidades están vacías: vale lo que declara el backend.
    app, jobs = _app(capable=False)
    resp, _ = _post(app, {"model": "ltx-2.5", "prompt": "x"})
    assert resp.status_code == 200, resp.text
    jobs.submit.assert_awaited_once()


def test_un_modelo_que_no_genera_video_da_400():
    app, jobs = _app(capable=False, backend_type="llamacpp_mks")
    resp, _ = _post(app, {"model": "ltx-2.5", "prompt": "x"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "model_not_capable"


def test_un_modelo_inexistente_da_404():
    app, _ = _app(profile=False)
    app.state.model_manager.get_state = AsyncMock(return_value=None)
    resp, _ = _post(app, {"model": "no-existe", "prompt": "x"})
    assert resp.status_code == 404


# ── Imagen a vídeo y primer/último fotograma ───────────────────────────


def _png(width: int, height: int) -> bytes:
    import struct
    import zlib

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = b"IHDR" + ihdr
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk))
    )


def test_lee_las_dimensiones_de_png_jpeg_y_webp():
    from ocabra.core.image_header import image_size

    assert image_size(_png(1000, 500)) == (1000, 500)
    jpeg = (
        b"\xff\xd8"
        + b"\xff\xe0\x00\x10" + b"JFIF\x00" + b"\x00" * 9
        + b"\xff\xc0\x00\x11\x08" + (480).to_bytes(2, "big") + (640).to_bytes(2, "big")
        + b"\x03" + b"\x00" * 9
    )  # fmt: skip
    assert image_size(jpeg) == (640, 480)
    webp = (
        b"RIFF\x00\x00\x00\x00WEBPVP8X"
        + b"\x00" * 8
        + (799).to_bytes(3, "little")
        + (599).to_bytes(3, "little")
    )
    assert image_size(webp) == (800, 600)
    assert image_size(b"esto no es una imagen, solo texto largo") is None


def _post_multipart(app, data, files):
    session = _Session()
    with patch(f"{MODULE}.AsyncSessionLocal", lambda: session):
        resp = TestClient(app, raise_server_exceptions=False).post(
            "/v1/videos", data=data, files=files
        )
    return resp


def test_input_reference_por_multipart_anima_la_imagen_con_su_proporcion():
    import base64

    app, jobs = _app()
    image = _png(1000, 1000)
    resp = _post_multipart(
        app,
        {"model": "ltx-2.5", "prompt": "que se mueva"},
        {"input_reference": ("foto.png", image, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["size"] == "960x960"
    params = jobs.submit.await_args.args[0].params
    assert params["mode"] == "i2v"
    assert base64.b64decode(params["image_b64"]) == image
    assert "last_image_b64" not in params


def test_primer_y_ultimo_fotograma_por_data_url():
    import base64

    app, jobs = _app()
    first = "data:image/png;base64," + base64.b64encode(_png(1280, 720)).decode()
    last = {"image_url": "data:image/png;base64," + base64.b64encode(_png(1280, 720)).decode()}
    resp, _ = _post(
        app,
        {"model": "ltx-2.5", "prompt": "x", "input_reference": first, "last_frame": last},
    )
    assert resp.status_code == 200, resp.text
    params = jobs.submit.await_args.args[0].params
    assert params["mode"] == "flf2v"
    assert params["image_b64"] and params["last_image_b64"]
    assert resp.json()["size"] == "1280x704"


@pytest.mark.parametrize(
    ("extra", "param"),
    [
        ({"last_frame": "data:image/png;base64," + __import__("base64").b64encode(
            _png(64, 64)).decode()}, "last_frame"),
        ({"input_reference": "data:text/plain;base64,aG9sYQ=="}, "input_reference"),
        ({"input_reference": "ftp://example.com/x.png"}, "input_reference"),
    ],
)  # fmt: skip
def test_rechaza_imagenes_invalidas(extra, param):
    app, jobs = _app()
    resp, _ = _post(app, {"model": "ltx-2.5", "prompt": "x", **extra})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error"]["param"] == param
    jobs.submit.assert_not_awaited()


def test_los_workflows_de_imagen_rellenan_sus_imagenes():
    worker = _worker_module()
    base = WORKERS / "comfyui_video_workflows"
    i2v = worker.WorkflowTemplate(base / "ltx25_i2v.json")
    assert i2v.image_inputs == ["image"]
    assert len(i2v.samplers) == 2
    prompt, _ = i2v.render(
        {"prompt": "x", "width": 960, "height": 960, "seconds": 3}, "p", {"image": "j_image"}
    )
    assert prompt[i2v.params["image"]]["inputs"]["image"] == "j_image"

    flf = worker.WorkflowTemplate(base / "ltx25_flf2v.json")
    assert flf.image_inputs == ["image", "last_image"]
    # Una sola pasada: todo el progreso de muestreo en un tramo.
    assert len(flf.samplers) == 1 and len(flf.stage_weights) == 1
    prompt, _ = flf.render(
        {"prompt": "x", "width": 1280, "height": 704, "seconds": 3},
        "p",
        {"image": "a", "last_image": "b"},
    )
    assert prompt[flf.params["image"]]["inputs"]["image"] == "a"
    assert prompt[flf.params["last_image"]]["inputs"]["image"] == "b"
    assert prompt[flf.params["width"]]["inputs"]["value"] == 1280


def test_el_backend_da_los_tres_modos_por_defecto():
    from ocabra.backends.comfyui_video_backend import ComfyUIVideoBackend

    paths = ComfyUIVideoBackend.workflow_paths({})
    assert sorted(paths) == ["flf2v", "i2v", "t2v"]
    with pytest.raises(ValueError):
        ComfyUIVideoBackend.workflow_paths({"workflows": {"v2v": "ltx25_t2v"}})


# ── Tiempos y velocidad en las estadísticas ────────────────────────────


def test_la_velocidad_de_un_video_son_los_fps_generados():
    from ocabra.stats.aggregator import video_speed, work_speeds

    row = SimpleNamespace(
        request_kind="video_generation",
        error=None,
        status_code=200,
        duration_ms=90_000,
        runtime_telemetry={"queue_ms": 10_000, "load_ms": 20_000},
        work_size_meta={"input_frames": 121, "generation_fps": 2.017},
    )
    speeds = work_speeds([row])
    fps = next(s for s in speeds if s["unit"] == "frames_per_s")
    # 121 fotogramas en 60 s: sin la cola ni la carga.
    assert fps["value"] == round(121 / 60, 2)
    assert video_speed(row.work_size_meta) == (2.02, 121)
    assert video_speed({"audio_seconds": 3}) == (None, None)


@pytest.mark.asyncio
async def test_un_trabajo_registra_cola_carga_y_generacion_por_separado():
    import time as _time
    import uuid as _uuid

    from ocabra.core.request_telemetry import RequestRuntimeTelemetry
    from ocabra.core.video_jobs import PendingVideo, VideoJobManager

    state = SimpleNamespace(
        model_id="comfyui_video/video",
        backend_type="comfyui_video",
        backend_model_id="video",
        capabilities=SimpleNamespace(video_generation=True),
        current_gpu=[1],
    )
    backend = MagicMock()

    async def _generate(*_a, **_k):
        await __import__("asyncio").sleep(0.05)
        return b"MP4", {"x-ocabra-generation-s": "60"}

    backend.generate_video = _generate
    backend.get_progress = AsyncMock(return_value=None)
    app = SimpleNamespace(
        state=SimpleNamespace(
            model_manager=MagicMock(begin_request=MagicMock(return_value="r1")),
            worker_pool=MagicMock(get_backend=AsyncMock(return_value=backend)),
            profile_registry=MagicMock(),
            router_resolver=None,
        )
    )

    class _S:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, *_a, **_k):
            return None

        async def commit(self):
            return None

    manager = VideoJobManager(app, lambda: _S())
    request = SimpleNamespace(state=SimpleNamespace())
    pending = PendingVideo(
        job_id=_uuid.uuid4(),
        model="ltx-2.5",
        params={"prompt": "x", "width": 1280, "height": 704, "seconds": 5, "fps": 24},
        user=None,
        request=request,
    )
    pending.submitted_at = _time.monotonic() - 3.0  # 3 s en cola

    recorded = {}

    async def _record(req, **kwargs):
        recorded.update(kwargs, telemetry=req.state.runtime_telemetry)

    with (
        patch(
            "ocabra.core.inference.resolution.resolve_profile",
            AsyncMock(return_value=(SimpleNamespace(), state)),
        ),
        patch("ocabra.stats.collector.record_direct_request", _record),
        patch("ocabra.core.video_jobs.videos_dir", lambda: Path(__import__("tempfile").mkdtemp())),
    ):
        await manager._run(pending)

    telemetry = recorded["telemetry"]
    assert isinstance(telemetry, RequestRuntimeTelemetry)
    assert telemetry.queue_ms >= 3000
    assert telemetry.load_ms is not None and telemetry.gpu_indices == [1]
    assert recorded["generation_ms"] is not None
    assert recorded["duration_ms"] >= 3000  # cuenta desde el POST
    assert recorded["started_at"] == pending.submitted_wall
    meta = recorded["work_size_meta"]
    assert meta["input_frames"] == 121 and meta["generation_fps"] > 0


# ── Nombre del modelo de vídeo independiente de la carpeta ─────────────


@pytest.mark.asyncio
async def test_el_alta_de_video_se_llama_como_el_modelo_y_guarda_la_carpeta(tmp_path):
    from ocabra.api.internal import registry as reg

    (tmp_path / "video" / "diffusion_models").mkdir(parents=True)
    added = {}

    async def _add_model(**kwargs):
        added.update(kwargs)
        return SimpleNamespace(model_id=kwargs["model_id"], to_dict=lambda: {"ok": True})

    mm = MagicMock(get_state=AsyncMock(return_value=None), add_model=_add_model)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(model_manager=mm, profile_registry=None))
    )
    body = reg.RegisterLocalModelRequest(
        model_ref="ltx-2.5", path=str(tmp_path / "video"), backend_type="comfyui_video"
    )
    with (
        patch.object(reg.settings, "models_dir", str(tmp_path)),
        patch.object(reg.settings, "ollama_shared_models_dir", ""),
    ):
        await reg.register_local_model(body, request, _user=None)

    assert added["model_id"] == "comfyui_video/ltx-2.5"
    assert added["extra_config"]["comfyui_video"]["model_dir"] == "video"


@pytest.mark.asyncio
async def test_borrar_un_modelo_de_video_no_toca_la_carpeta_compartida(tmp_path):
    from ocabra.api.internal import models as models_api

    folder = tmp_path / "video"
    (folder / "diffusion_models").mkdir(parents=True)
    with patch.object(models_api.settings, "models_dir", str(tmp_path)):
        deleted = await models_api._delete_model_files(
            "comfyui_video/video", "comfyui_video", {"comfyui_video": {"model_dir": "video"}}
        )
    assert deleted is None
    assert (folder / "diffusion_models").is_dir()
