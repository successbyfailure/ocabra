"""Regresión del worker de escalado.

Lo importante que se prueba aquí es la conversión RGB→YUV420p hecha a mano en
la GPU: es la optimización que triplica el rendimiento (de 19 a 63 fps
medidos), pero una matriz mal puesta se traduce en un cambio de color en todos
los vídeos, y eso no lo detecta ningún test de humo.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location(
    "upscaler_worker",
    Path(__file__).resolve().parents[1] / "workers" / "upscaler_worker.py",
)
worker = importlib.util.module_from_spec(_SPEC)
sys.modules["upscaler_worker"] = worker
_SPEC.loader.exec_module(worker)


def _yuv_planes(rgb: "torch.Tensor", h: int, w: int):
    raw = worker._rgb_to_yuv420p(rgb)
    y_size, c_size = h * w, (h // 2) * (w // 2)
    return raw[:y_size], raw[y_size : y_size + c_size], raw[y_size + c_size :]


@pytest.mark.parametrize(
    "color, luma_esperada, cromas_neutras",
    [
        # BT.709 rango limitado: negro=16, blanco=235, y croma neutra en 128.
        ((0.0, 0.0, 0.0), 16, True),
        ((1.0, 1.0, 1.0), 235, True),
        ((0.5, 0.5, 0.5), 126, True),
    ],
)
def test_grises_dan_luma_correcta_y_croma_neutra(color, luma_esperada, cromas_neutras):
    h = w = 4
    rgb = torch.zeros(1, 3, h, w)
    for canal, valor in enumerate(color):
        rgb[:, canal] = valor
    luma, cb, cr = _yuv_planes(rgb, h, w)
    assert abs(luma[0] - luma_esperada) <= 1, f"luma {luma[0]} != {luma_esperada}"
    if cromas_neutras:
        assert abs(cb[0] - 128) <= 1
        assert abs(cr[0] - 128) <= 1


def test_rojo_puro_desplaza_cr_y_azul_desplaza_cb():
    """Comprobación de signo: confundir Cb y Cr intercambia rojo y azul."""
    h = w = 4
    rojo = torch.zeros(1, 3, h, w)
    rojo[:, 0] = 1.0
    azul = torch.zeros(1, 3, h, w)
    azul[:, 2] = 1.0
    _, cb_rojo, cr_rojo = _yuv_planes(rojo, h, w)
    _, cb_azul, cr_azul = _yuv_planes(azul, h, w)
    # Se comprueba la dirección respecto al neutro (128), no valores absolutos:
    # en BT.709 el rojo puro da Cr=240 pero Cb=102, y el azul Cb=240 y Cr=118,
    # así que un umbral fijo tipo "<100" fallaría siendo la matriz correcta.
    assert cr_rojo[0] > 128 and cb_rojo[0] < 128, "el rojo debe subir Cr y bajar Cb"
    assert cb_azul[0] > 128 and cr_azul[0] < 128, "el azul debe subir Cb y bajar Cr"
    assert cr_rojo[0] > cr_azul[0] and cb_azul[0] > cb_rojo[0]


def test_tamano_del_plano_es_420():
    h, w = 8, 8
    rgb = torch.rand(1, 3, h, w)
    raw = worker._rgb_to_yuv420p(rgb)
    assert len(raw) == h * w + 2 * (h // 2) * (w // 2)


def test_arquitectura_acepta_los_pesos_oficiales():
    """El state_dict oficial de Real-ESRGAN Compact debe encajar entero."""
    model = worker.SRVGGNetCompact()
    claves = set(model.state_dict())
    # 1 conv inicial + 32 bloques (conv+prelu) + 1 conv final, con pesos y sesgos.
    assert len(claves) == 101, len(claves)
