"""Which CUDA runtime llama-server workers get on their LD_LIBRARY_PATH."""

import pytest

from ocabra.backends import llama_cpp_backend as lcb


@pytest.fixture(autouse=True)
def _fresh_cache():
    lcb._system_cuda_runtime_available.cache_clear()
    yield
    lcb._system_cuda_runtime_available.cache_clear()


def _venvs(monkeypatch, available: dict[str, str]):
    monkeypatch.setattr(
        lcb, "venv_nvidia_ld_library_path", lambda _dir, backend: available.get(backend, "")
    )


def test_system_runtime_wins_over_another_backends_venv(monkeypatch) -> None:
    """2026-11-10: workers ran on vLLM's cuBLAS although the image ships the
    CUDA toolkit llama-server is built with."""
    _venvs(monkeypatch, {"vllm": "/data/backends/vllm/venv/nvidia/cu13/lib"})
    monkeypatch.setattr(lcb.ctypes.util, "find_library", lambda name: f"lib{name}.so.13")

    assert lcb._cuda_ld_library_path() == ""


def test_borrows_a_venv_runtime_only_when_the_system_has_none(monkeypatch) -> None:
    _venvs(monkeypatch, {"vllm": "/data/backends/vllm/venv/nvidia/cu13/lib"})
    monkeypatch.setattr(
        lcb.ctypes.util, "find_library", lambda name: None if name == "cublas" else f"lib{name}.so"
    )

    assert lcb._cuda_ld_library_path() == "/data/backends/vllm/venv/nvidia/cu13/lib"


def test_llama_cpp_own_venv_still_comes_first(monkeypatch) -> None:
    _venvs(monkeypatch, {"llama_cpp": "/own", "vllm": "/vllm"})
    monkeypatch.setattr(lcb.ctypes.util, "find_library", lambda name: f"lib{name}.so")

    assert lcb._cuda_ld_library_path() == "/own"
