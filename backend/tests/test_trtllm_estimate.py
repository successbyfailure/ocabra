import sys
from types import SimpleNamespace

from starlette.datastructures import QueryParams

from ocabra.api.internal import trtllm


def test_estimate_warnings_use_selected_gpu_indices(monkeypatch):
    class FakePynvml:
        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlShutdown():
            return None

        @staticmethod
        def nvmlDeviceGetCount():
            return 2

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):
            return index

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            total_mb = 24 * 1024 if handle == 1 else 12 * 1024
            free_mb = 20 * 1024 if handle == 1 else 2 * 1024
            return SimpleNamespace(total=total_mb * 1024**2, free=free_mb * 1024**2)

    monkeypatch.setitem(sys.modules, "pynvml", FakePynvml)

    warnings = trtllm._estimate_warnings(
        build_mb=10 * 1024,
        serve_mb=8 * 1024,
        tp_size=1,
        gpu_indices=[1],
        params_b=32,
    )

    assert warnings == []


def test_estimate_warnings_report_selected_gpu_when_it_lacks_memory(monkeypatch):
    class FakePynvml:
        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlShutdown():
            return None

        @staticmethod
        def nvmlDeviceGetCount():
            return 2

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):
            return index

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            total_mb = 24 * 1024
            free_mb = 5 * 1024 if handle == 1 else 20 * 1024
            return SimpleNamespace(total=total_mb * 1024**2, free=free_mb * 1024**2)

    monkeypatch.setitem(sys.modules, "pynvml", FakePynvml)

    warnings = trtllm._estimate_warnings(
        build_mb=10 * 1024,
        serve_mb=8 * 1024,
        tp_size=1,
        gpu_indices=[1],
        params_b=32,
    )

    assert warnings == ["Build probablemente no cabe: necesita ~10.0GB/GPU y la GPU 1 tiene 5.0GB libres ahora"]


def test_estimate_warnings_report_tight_margin_more_clearly(monkeypatch):
    class FakePynvml:
        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlShutdown():
            return None

        @staticmethod
        def nvmlDeviceGetCount():
            return 2

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):
            return index

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            total_mb = 24 * 1024
            free_mb = 23853 if handle == 1 else 11575
            return SimpleNamespace(total=total_mb * 1024**2, free=free_mb * 1024**2)

    monkeypatch.setitem(sys.modules, "pynvml", FakePynvml)

    warnings = trtllm._estimate_warnings(
        build_mb=24576,
        serve_mb=8 * 1024,
        tp_size=1,
        gpu_indices=[1],
        params_b=32,
    )

    assert warnings == [
        "Muy justo: build necesita ~24.0GB/GPU y la GPU 1 tiene 23.3GB libres ahora (723 MB por debajo). Puede funcionar si liberas algo de VRAM antes de empezar."
    ]


def test_parse_gpu_indices_query_supports_single_and_csv_values():
    assert trtllm._parse_gpu_indices_query(SimpleNamespace(query_params=QueryParams("gpu_indices=1"))) == [1]
    assert trtllm._parse_gpu_indices_query(SimpleNamespace(query_params=QueryParams("gpu_indices=1&gpu_indices=0"))) == [1, 0]
    assert trtllm._parse_gpu_indices_query(SimpleNamespace(query_params=QueryParams("gpu_indices=1,0"))) == [1, 0]
