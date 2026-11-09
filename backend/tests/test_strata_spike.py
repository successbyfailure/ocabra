"""El spike de Strata mantiene comandos y resultados reproducibles."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import strata_spike  # noqa: E402


def test_plan_3090_aisla_la_3060(tmp_path: Path) -> None:
    command = strata_spike.build_launch_command(
        Path("/opt/strata"), tmp_path, profile="3090", port=18095, plan=True
    )

    assert command[command.index("--gpu-roles") + 1] == "1=main,0=off"
    assert command[command.index("--data-dir") + 1] == str(tmp_path / "models")
    assert command[-1] == "--plan"
    assert "--yes" not in command
    assert "--restart" not in command


def test_ejecucion_split_usa_3090_principal_y_3060_helper(tmp_path: Path) -> None:
    command = strata_spike.build_launch_command(
        Path("/opt/strata"), tmp_path, profile="split", port=18096, plan=False
    )

    assert command[command.index("--gpu-roles") + 1] == "1=main,0=helper"
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--vision") + 1] == "off"
    assert command[-2:] == ["--yes", "--restart"]


def test_perfil_desconocido_se_rechaza(tmp_path: Path) -> None:
    with pytest.raises(strata_spike.SpikeError, match="Perfil desconocido"):
        strata_spike.build_launch_command(
            Path("/opt/strata"), tmp_path, profile="otra", port=18095, plan=True
        )


def test_stream_openai_extrae_ttft_texto_y_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter((10.1,))
    monkeypatch.setattr(strata_spike.time, "monotonic", lambda: next(ticks))
    stream = io.BytesIO(
        b'data: {"choices":[{"delta":{"reasoning_content":"pienso"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"pong"}}]}\n\n'
        b'data: {"choices":[],"usage":{"completion_tokens":3}}\n\n'
        b"data: [DONE]\n\n"
    )

    ttft, text, tokens = strata_spike._read_sse(stream, 10.0)

    assert ttft == pytest.approx(0.1)
    assert text == "pong"
    assert tokens == 3
