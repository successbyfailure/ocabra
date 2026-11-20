"""Tamaño del trabajo de un escalado de foto en las estadísticas (20/11).

Solo se guardaba ``n_images: 1``: una foto de 1080p (~10 s) y una 4K (~45 s)
quedaban igual, y el estimador no podía relacionar tiempo con píxeles.
"""

from __future__ import annotations

from ocabra.api.openai.image_upscale import upscale_work_meta
from ocabra.core.duration_estimator import extract_work_units


def test_worker_headers_give_the_size_of_the_work():
    meta = upscale_work_meta(
        {
            "X-Ocabra-Input-Resolution": "1920x1080",
            "X-Ocabra-Model-Resolution": "7680x4320",
            "X-Ocabra-Delivered-Resolution": "3840x2160",
            "X-Ocabra-Tiles": "12",
            "X-Ocabra-Seconds": "9.1",
        }
    )
    assert meta == {
        "input_w": 1920, "input_h": 1080, "input_mp": 2.074,
        "model_w": 7680, "model_h": 4320, "model_mp": 33.178,
        "output_w": 3840, "output_h": 2160, "output_mp": 8.294,
        "tiles": 12,
    }


def test_missing_or_bad_headers_leave_nothing():
    assert upscale_work_meta({"x-ocabra-input-resolution": "nope", "x-ocabra-tiles": "x"}) == {}


def test_the_estimator_counts_a_photo_as_one_frame_of_its_model_output():
    assert extract_work_units("upscaler", {"n_images": 1, "model_mp": 33.178}) == 33.178
    # Video keeps frames × output megapixels.
    assert extract_work_units(
        "upscaler", {"input_frames": 10, "output_w": 1920, "output_h": 1080}
    ) == 10 * 1920 * 1080 / 1_000_000
