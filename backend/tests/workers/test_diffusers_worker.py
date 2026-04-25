"""Tests for the diffusers worker pipeline detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def worker_module(monkeypatch: pytest.MonkeyPatch):
    """Import the worker module without dragging the runtime stack.

    The worker imports ``uvicorn``, ``fastapi`` and ``pydantic`` at the top
    level — none of those are needed for the pure-helper tests in this
    module. Stubbing them via ``sys.modules`` lets us exercise
    ``detect_pipeline_class`` from a minimal env (the worker venv has them,
    the test sandbox does not).
    """
    import types

    def _stub(name: str, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules.setdefault(name, mod)

    class _Stub:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

    _stub("uvicorn", run=lambda *a, **k: None)
    _stub(
        "fastapi",
        FastAPI=_Stub,
        HTTPException=type("HTTPException", (Exception,), {}),
    )
    _stub(
        "pydantic",
        BaseModel=_Stub,
        ConfigDict=lambda **kw: kw,
    )

    workers_root = Path(__file__).resolve().parents[2] / "workers"
    monkeypatch.syspath_prepend(str(workers_root))
    import importlib

    if "diffusers_worker" in sys.modules:
        del sys.modules["diffusers_worker"]
    return importlib.import_module("diffusers_worker")


# ---------------------------------------------------------------------------
# detect_pipeline_class — HuggingFace tree (existing behaviour)
# ---------------------------------------------------------------------------


def test_detect_pipeline_class_reads_model_index_json(
    tmp_path: Path, worker_module
) -> None:
    model_dir = tmp_path / "stabilityai--sdxl-turbo"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"})
    )

    assert worker_module.detect_pipeline_class(model_dir) == "StableDiffusionXLPipeline"


def test_detect_pipeline_class_rejects_dir_without_index(
    tmp_path: Path, worker_module
) -> None:
    model_dir = tmp_path / "broken"
    model_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        worker_module.detect_pipeline_class(model_dir)


# ---------------------------------------------------------------------------
# detect_pipeline_class — single-file checkpoints (new path)
# ---------------------------------------------------------------------------


def test_detect_pipeline_class_safetensors_sdxl_by_name(
    tmp_path: Path, worker_module
) -> None:
    """SDXL is inferred from filename tokens (sd_xl, sdxl, xl-base)."""
    for stem in ("sd_xl_base_1.0", "sdxl_turbo", "stable-diffusion-xl-base", "model-xl-base"):
        ckpt = tmp_path / f"{stem}.safetensors"
        ckpt.write_text("not a real checkpoint")
        assert (
            worker_module.detect_pipeline_class(ckpt)
            == "StableDiffusionXLPipeline"
        ), f"failed to detect SDXL for {stem}"


def test_detect_pipeline_class_safetensors_default_sd15(
    tmp_path: Path, worker_module
) -> None:
    """Names that don't match SDXL tokens fall back to SD 1.5."""
    ckpt = tmp_path / "v1-5-pruned-emaonly.safetensors"
    ckpt.write_text("placeholder")
    assert worker_module.detect_pipeline_class(ckpt) == "StableDiffusionPipeline"


def test_detect_pipeline_class_ckpt_extension_supported(
    tmp_path: Path, worker_module
) -> None:
    ckpt = tmp_path / "sd_xl_base_1.0.ckpt"
    ckpt.write_text("placeholder")
    assert worker_module.detect_pipeline_class(ckpt) == "StableDiffusionXLPipeline"


def test_detect_pipeline_class_rejects_unknown_extension(
    tmp_path: Path, worker_module
) -> None:
    bogus = tmp_path / "model.bin"
    bogus.write_text("placeholder")
    with pytest.raises(ValueError, match="single-file checkpoint extension"):
        worker_module.detect_pipeline_class(bogus)


def test_detect_pipeline_class_env_override(
    tmp_path: Path, worker_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DIFFUSERS_PIPELINE_OVERRIDE forces the class even for ambiguous names."""
    monkeypatch.setenv("DIFFUSERS_PIPELINE_OVERRIDE", "FluxPipeline")
    ckpt = tmp_path / "some_random_checkpoint.safetensors"
    ckpt.write_text("placeholder")
    assert worker_module.detect_pipeline_class(ckpt) == "FluxPipeline"


def test_detect_pipeline_class_env_override_validates_class(
    tmp_path: Path, worker_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIFFUSERS_PIPELINE_OVERRIDE", "BogusPipeline")
    ckpt = tmp_path / "x.safetensors"
    ckpt.write_text("placeholder")
    with pytest.raises(ValueError, match="DIFFUSERS_PIPELINE_OVERRIDE"):
        worker_module.detect_pipeline_class(ckpt)
