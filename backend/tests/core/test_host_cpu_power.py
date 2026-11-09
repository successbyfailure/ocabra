"""Escala de la barra de potencia de la CPU (dashboard)."""

from __future__ import annotations

import pytest

from ocabra.api.internal import host


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("AMD Ryzen Threadripper 3960X 24-Core Processor", (280.0, "tdp")),
        ("AMD Ryzen Threadripper PRO 3975WX 32-Cores", (280.0, "tdp")),
        ("AMD Ryzen Threadripper 7980X 64-Cores", (350.0, "tdp")),
        ("Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz", (None, None)),
        (None, (None, None)),
    ],
)
def test_cpu_power_max_from_known_tdp(monkeypatch, model, expected):
    monkeypatch.setattr(host, "_rapl_limit_w", lambda: None)
    monkeypatch.setattr("ocabra.config.settings.cpu_power_max_w", None)
    assert host.cpu_power_max(model) == expected


def test_config_and_rapl_take_precedence(monkeypatch):
    model = "AMD Ryzen Threadripper 3960X 24-Core Processor"
    monkeypatch.setattr(host, "_rapl_limit_w", lambda: 250.0)
    monkeypatch.setattr("ocabra.config.settings.cpu_power_max_w", None)
    assert host.cpu_power_max(model) == (250.0, "rapl")
    monkeypatch.setattr("ocabra.config.settings.cpu_power_max_w", 300.0)
    assert host.cpu_power_max(model) == (300.0, "config")
