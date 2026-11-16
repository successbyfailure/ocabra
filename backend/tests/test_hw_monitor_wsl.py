"""Characterization tests for hw-monitor's reduced NVML surface on WSL2."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pynvml


def _load_hw_monitor() -> ModuleType:
    path = Path(__file__).parents[2] / "hw-monitor" / "hw_monitor.py"
    spec = importlib.util.spec_from_file_location("ocabra_hw_monitor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_read_gpu_preserves_vram_when_wsl_omits_optional_metrics() -> None:
    monitor = _load_hw_monitor()
    mib = 1024 * 1024
    memory = SimpleNamespace(total=12_000 * mib, free=10_000 * mib, used=2_000 * mib)

    with (
        patch.object(monitor.pynvml, "nvmlDeviceGetHandleByIndex", return_value=MagicMock()),
        patch.object(monitor.pynvml, "nvmlDeviceGetMemoryInfo", return_value=memory),
        patch.object(
            monitor.pynvml,
            "nvmlDeviceGetUtilizationRates",
            side_effect=pynvml.NVMLError_NotSupported(),
        ),
        patch.object(
            monitor.pynvml,
            "nvmlDeviceGetTemperature",
            side_effect=pynvml.NVMLError_NotSupported(),
        ),
        patch.object(
            monitor.pynvml,
            "nvmlDeviceGetPowerUsage",
            side_effect=pynvml.NVMLError_NotSupported(),
        ),
        patch.object(monitor.pynvml, "nvmlDeviceGetName", return_value=b"RTX 3090"),
        patch.object(monitor, "_read_gpu_processes", return_value=[]),
    ):
        state = monitor._read_gpu(0)

    assert state["free_vram_mb"] == 10_000
    assert state["used_vram_mb"] == 2_000
    assert state["utilization_pct"] == 0.0
    assert state["temperature_c"] == 0.0
    assert state["power_draw_w"] == 0.0
