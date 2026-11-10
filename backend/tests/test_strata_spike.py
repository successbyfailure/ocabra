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


def test_qwen38_split_engine_fija_familia_modelo_y_mtp(tmp_path: Path) -> None:
    command = strata_spike.build_launch_command(
        Path("/opt/strata"),
        tmp_path,
        target="qwen38",
        profile="split-engine",
        port=18097,
        plan=True,
    )

    assert command[command.index("--family") + 1] == "qwen"
    assert command[command.index("--model") + 1] == "Q2_0"
    assert command[command.index("--drafting") + 1] == "mtp"
    assert command[command.index("--split-engine") + 1] == "on"
    assert command[command.index("--gpu-roles") + 1] == "1=main,0=helper"


def test_espacio_requerido_suma_reserva_a_la_descarga() -> None:
    plan = {"plan": {"download": {"gb": 67.6}}}

    assert strata_spike._required_free_disk_gib(plan) == pytest.approx(77.6)
    assert strata_spike._required_free_disk_gib({"plan": {"download": {"gb": 0}}}) == 45


def test_perfil_desconocido_se_rechaza(tmp_path: Path) -> None:
    with pytest.raises(strata_spike.SpikeError, match="Perfil desconocido"):
        strata_spike.build_launch_command(
            Path("/opt/strata"), tmp_path, profile="otra", port=18095, plan=True
        )


def test_entorno_strata_aisla_componentes_estado_y_credencial(tmp_path: Path) -> None:
    environment = strata_spike._strata_environment(tmp_path, "secreta")

    assert environment["STRATA_SPIKE_API_KEY"] == "secreta"
    assert environment["STRATA_DATA_DIR"] == str(tmp_path / "models")
    assert environment["XDG_DATA_HOME"] == str(tmp_path / "runtime" / "xdg-data")
    assert environment["XDG_STATE_HOME"] == str(tmp_path / "runtime" / "xdg-state")


def test_log_del_engine_se_archiva_acotado(tmp_path: Path) -> None:
    log_root = tmp_path / "runtime" / "xdg-state" / "strata" / "launch"
    log_root.mkdir(parents=True)
    log = log_root / "q2_0.engine.log"
    log.write_text("uno\ndos\ntres\n", encoding="utf-8")

    evidence = strata_spike._latest_engine_log(tmp_path, lines=2)

    assert evidence == {"path": str(log), "tail": ["dos", "tres"]}


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
