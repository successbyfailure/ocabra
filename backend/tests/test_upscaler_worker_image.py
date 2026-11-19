"""Ruta de fotos del worker de escalado (``/upscale/image``) con un modelo falso.

Se prueba sin GPU: el worker coloca la foto en el dispositivo de los pesos del
modelo, así que un modelo en CPU basta para cubrir el teselado, la
transparencia, los formatos de salida y los límites.
"""

import importlib.util
import io
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from fastapi.testclient import TestClient  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "upscaler_worker_image",
    Path(__file__).resolve().parents[1] / "workers" / "upscaler_worker.py",
)
worker = importlib.util.module_from_spec(_SPEC)
sys.modules["upscaler_worker_image"] = worker
_SPEC.loader.exec_module(worker)


class NearestX2(torch.nn.Module):
    """x2 por vecino más cercano: sin campo receptivo, salida exacta."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest") + self.w


class BlurX2(torch.nn.Module):
    """Media 3x3 y luego x2: tiene contexto, así que sin margen se verían costuras."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1 / 9)

    def forward(self, x):
        return torch.nn.functional.interpolate(self.conv(x), scale_factor=2, mode="nearest")


def _state(model, scale=2):
    state = worker.WorkerState("upscaler/fake", Path("/nonexistent"))
    state.model = model.eval()
    state.scale = scale
    state.half = False
    return state


def _png(mode="RGB", size=(10, 6), color=(10, 200, 30)):
    if mode == "RGBA":
        color = (*color, 128)
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def _random_rgb(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_teselas_dan_lo_mismo_que_una_pasada():
    rgb = _random_rgb(37, 23)
    state = _state(NearestX2())
    full, n_full = worker.upscale_rgb_tiled(state, rgb, tile=1000, pad=0)
    tiled, n_tiled = worker.upscale_rgb_tiled(state, rgb, tile=16, pad=4)
    assert n_full == 1 and n_tiled == 6
    assert full.shape == (74, 46, 3)
    assert np.array_equal(full, tiled)
    assert np.array_equal(full[::2, ::2], rgb)


def test_el_margen_evita_costuras_con_un_modelo_con_contexto():
    rgb = _random_rgb(40, 40, seed=3)
    state = _state(BlurX2())
    full, _ = worker.upscale_rgb_tiled(state, rgb, tile=1000, pad=0)
    con_margen, _ = worker.upscale_rgb_tiled(state, rgb, tile=16, pad=2)
    sin_margen, _ = worker.upscale_rgb_tiled(state, rgb, tile=16, pad=0)
    diff = np.abs(full.astype(int) - con_margen.astype(int))
    assert diff.max() <= 1
    assert np.abs(full.astype(int) - sin_margen.astype(int)).max() > 1


@pytest.mark.parametrize(
    "kwargs, esperado",
    [
        ({}, (40, 24)),
        ({"scale": 2}, (20, 12)),
        ({"scale": 8}, (40, 24)),  # nunca por encima de lo que da el modelo
        ({"target_height": 12}, (20, 12)),
        ({"target_width": 30, "target_height": 1000}, (30, 18)),
        ({"target_height": 5000}, (40, 24)),
    ],
)
def test_plan_del_tamano_entregado(kwargs, esperado):
    assert worker.plan_delivered_size(10, 6, 4, **kwargs) == esperado


def test_png_por_defecto_con_estadisticas():
    body, media_type, stats = worker.upscale_image_sync(_state(NearestX2()), _png())
    out = Image.open(io.BytesIO(body))
    assert media_type == "image/png"
    assert out.format == "PNG" and out.size == (20, 12) and out.mode == "RGB"
    assert out.getpixel((5, 5)) == (10, 200, 30)
    assert stats["input_resolution"] == "10x6"
    assert stats["model_resolution"] == "20x12"
    assert stats["delivered_resolution"] == "20x12"
    assert stats["delivered_height"] == 12
    assert stats["alpha"] == 0


def test_la_transparencia_se_conserva_en_png_y_webp_y_se_pierde_en_jpeg():
    state = _state(NearestX2())
    for fmt in ("png", "webp"):
        body, _, stats = worker.upscale_image_sync(state, _png("RGBA"), output_format=fmt)
        out = Image.open(io.BytesIO(body))
        assert out.mode == "RGBA" and out.size == (20, 12)
        assert abs(out.getchannel("A").getpixel((3, 3)) - 128) <= 2
        assert stats["alpha"] == 1
    body, media_type, stats = worker.upscale_image_sync(state, _png("RGBA"), output_format="jpg")
    out = Image.open(io.BytesIO(body))
    assert media_type == "image/jpeg" and out.format == "JPEG" and out.mode == "RGB"
    assert stats["alpha"] == 0


def test_target_height_reduce_tras_el_modelo():
    body, _, stats = worker.upscale_image_sync(
        _state(NearestX2()), _png(size=(100, 60)), target_height=90
    )
    assert Image.open(io.BytesIO(body)).size == (150, 90)
    assert stats["model_resolution"] == "200x120"
    assert stats["delivered_resolution"] == "150x90"


def test_errores_de_entrada():
    state = _state(NearestX2())
    with pytest.raises(worker.ImageRequestError) as exc:
        worker.upscale_image_sync(state, b"esto no es una imagen")
    assert exc.value.status_code == 400
    with pytest.raises(worker.ImageRequestError):
        worker.upscale_image_sync(state, _png(), output_format="gif")


def test_demasiados_pixeles_da_413_sin_decodificar(monkeypatch):
    monkeypatch.setattr(worker, "IMAGE_MAX_PIXELS", 50)
    model = NearestX2()
    with pytest.raises(worker.ImageRequestError) as exc:
        worker.upscale_image_sync(_state(model), _png())
    assert exc.value.status_code == 413
    assert model.calls == 0


def test_webp_demasiado_grande_da_400(monkeypatch):
    monkeypatch.setitem(worker._FORMAT_MAX_SIDE, "webp", 15)
    with pytest.raises(worker.ImageRequestError, match="webp"):
        worker.upscale_image_sync(_state(NearestX2()), _png(), output_format="webp")


def test_sin_vram_se_reintenta_con_teselas_mas_pequenas():
    class OomOnBig(NearestX2):
        def forward(self, x):
            if x.shape[-1] * x.shape[-2] > 150 * 150:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
            return super().forward(x)

    # 256 → toda la foto (OOM), 128 → teselas de hasta 192 px con margen (OOM),
    # 64 → 128 px con margen: cabe.
    body, _, stats = worker.upscale_image_sync(_state(OomOnBig()), _png(size=(200, 200)), tile=256)
    assert Image.open(io.BytesIO(body)).size == (400, 400)
    assert stats["tiles"] == 16


def test_sin_vram_ni_con_teselas_minimas_da_507():
    class AlwaysOom(NearestX2):
        def forward(self, x):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(worker.ImageRequestError) as exc:
        worker.upscale_image_sync(_state(AlwaysOom()), _png())
    assert exc.value.status_code == 507


def test_ruta_http_del_worker():
    client = TestClient(worker.create_app(_state(NearestX2())))
    resp = client.post(
        "/upscale/image",
        params={"output_format": "webp", "scale": "1.5"},
        files={"file": ("p.png", _png(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/webp"
    assert resp.headers["x-ocabra-delivered-resolution"] == "15x9"
    assert resp.headers["x-ocabra-tiles"] == "1"
    assert Image.open(io.BytesIO(resp.content)).size == (15, 9)

    assert (
        client.post(
            "/upscale/image", params={"scale": "0"}, files={"file": ("p.png", _png(), "image/png")}
        ).status_code
        == 400
    )
    assert (
        client.post("/upscale/image", files={"file": ("p.png", b"nope", "image/png")}).status_code
        == 400
    )


def test_ruta_http_sin_modelo_da_503():
    state = worker.WorkerState("upscaler/fake", Path("/nonexistent"))
    resp = TestClient(worker.create_app(state)).post(
        "/upscale/image", files={"file": ("p.png", _png(), "image/png")}
    )
    assert resp.status_code == 503
