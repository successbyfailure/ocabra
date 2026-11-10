#!/usr/bin/env python3
"""Spike reproducible de Strata para el host 3060 + 3090 de oCabra.

El runner mantiene tres operaciones separadas:

* ``preflight`` no cambia nada y comprueba hardware, disco, procesos y version.
* ``plan`` pide a Strata su plan real de descarga/arranque, sin modificar nada.
* ``run`` necesita ``--execute``: arranca, mide y detiene siempre el servidor.

Los resultados se archivan bajo ``benchmark/results/strata-spike`` y nunca
incluyen la API key interna usada durante la ejecucion.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/docker/ai-models/ocabra/models/huggingface/strata-spike")
DEFAULT_RESULTS = PROJECT_ROOT / "benchmark" / "results" / "strata-spike"
DEFAULT_MODEL_FAMILY = "qwen36"
DEFAULT_MODEL = "UD-Q4_K_XL"
DEFAULT_CONTEXT = "32k"
DEFAULT_PORT = 18095
MIN_FREE_DISK_GIB = 45.0
MIN_TOTAL_RAM_GIB = 48.0

PROFILES = {
    "3090": "1=main,0=off",
    "split": "1=main,0=helper",
}

PROMPTS = (
    {
        "name": "short",
        "messages": [{"role": "user", "content": "Responde solo con la palabra pong."}],
        "max_tokens": 32,
    },
    {
        "name": "technical",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explica en espanol y en cinco puntos breves por que una cola FIFO "
                    "evita starvation en un servidor de inferencia."
                ),
            }
        ],
        "max_tokens": 256,
    },
)


class SpikeError(RuntimeError):
    """El spike no puede continuar de forma segura o reproducible."""


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    ttft_ms: float
    total_ms: float
    completion_tokens: int
    output_tokens_per_second: float
    text_preview: str


@dataclass(frozen=True)
class ToolResult:
    called: bool
    name: str | None
    arguments: object
    completion_tokens: int


def _run(command: list[str], *, timeout: int = 30, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SpikeError(f"Fallo ({completed.returncode}) en {' '.join(command)}: {detail}")
    return completed.stdout.strip()


def _strata_environment(root: Path, api_key: str) -> dict[str, str]:
    """Keep components, lifecycle journals and credentials inside the spike root."""
    runtime = root / "runtime"
    xdg_data = runtime / "xdg-data"
    xdg_state = runtime / "xdg-state"
    xdg_data.mkdir(parents=True, exist_ok=True)
    xdg_state.mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        "STRATA_SPIKE_API_KEY": api_key,
        "STRATA_DATA_DIR": str(root / "models"),
        "XDG_DATA_HOME": str(xdg_data),
        "XDG_STATE_HOME": str(xdg_state),
    }


def _strata_bin(root: Path, override: str | None = None) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
    else:
        path = root / "bin" / "strata"
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SpikeError(
            f"No existe un Terminal ejecutable en {path}. "
            "Ejecuta primero: benchmark/install_strata_spike.sh"
        )
    return path


def _memory() -> dict[str, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    gib = 1024**3
    return {
        "total_gib": round(values["MemTotal"] / gib, 2),
        "available_gib": round(values["MemAvailable"] / gib, 2),
        "swap_total_gib": round(values.get("SwapTotal", 0) / gib, 2),
        "swap_free_gib": round(values.get("SwapFree", 0) / gib, 2),
    }


def _gpus() -> list[dict[str, object]]:
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, object]] = []
    for line in raw.splitlines():
        index, uuid, name, memory_mb, driver = (part.strip() for part in line.split(",", 4))
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_mb": int(memory_mb),
                "driver": driver,
            }
        )
    return gpus


def _gpu_processes() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    if completed.returncode != 0:
        return []
    processes: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, name, memory_mb = (part.strip() for part in line.split(",", 3))
        processes.append(
            {"gpu_uuid": uuid, "pid": int(pid), "name": name, "memory_mb": int(memory_mb)}
        )
    return processes


def _gpu_usage() -> list[dict[str, object]]:
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, object]] = []
    for line in raw.splitlines():
        index, used, free, utilization, power = (part.strip() for part in line.split(",", 4))
        rows.append(
            {
                "index": int(index),
                "memory_used_mb": int(used),
                "memory_free_mb": int(free),
                "utilization_pct": int(utilization),
                "power_w": float(power),
            }
        )
    return rows


def _git_revision() -> dict[str, object]:
    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "status", "--short"]))
    return {"commit": commit, "dirty": dirty}


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def collect_preflight(root: Path, strata_bin: Path, port: int) -> dict[str, object]:
    disk = shutil.disk_usage(root)
    gib = 1024**3
    report: dict[str, object] = {
        "at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "strata": {
            "path": str(strata_bin),
            "version": _run([str(strata_bin), "--version"]),
        },
        "memory": _memory(),
        "disk": {
            "path": str(root),
            "free_gib": round(disk.free / gib, 2),
            "total_gib": round(disk.total / gib, 2),
        },
        "gpus": _gpus(),
        "gpu_processes": _gpu_processes(),
        "port": {"value": port, "free": _port_is_free(port)},
    }
    errors: list[str] = []
    gpu_map = {gpu["index"]: gpu for gpu in report["gpus"]}  # type: ignore[index]
    if 0 not in gpu_map or "3060" not in str(gpu_map[0]["name"]):
        errors.append("GPU 0 no es la RTX 3060 esperada")
    if 1 not in gpu_map or "3090" not in str(gpu_map[1]["name"]):
        errors.append("GPU 1 no es la RTX 3090 esperada")
    if float(report["memory"]["total_gib"]) < MIN_TOTAL_RAM_GIB:  # type: ignore[index]
        errors.append(f"RAM total inferior a {MIN_TOTAL_RAM_GIB:.0f} GiB")
    if float(report["disk"]["free_gib"]) < MIN_FREE_DISK_GIB:  # type: ignore[index]
        errors.append(f"espacio libre inferior a {MIN_FREE_DISK_GIB:.0f} GiB")
    if not report["port"]["free"]:  # type: ignore[index]
        errors.append(f"el puerto {port} esta ocupado")
    report["errors"] = errors
    blockers = []
    if report["gpu_processes"]:
        blockers.append("hay procesos de computo activos en las GPU objetivo")
    report["blockers"] = blockers
    report["ready_for_plan"] = not errors
    report["ready_for_run"] = not errors and not blockers
    report["ready"] = report["ready_for_run"]
    return report


def build_launch_command(
    strata_bin: Path,
    root: Path,
    *,
    profile: str,
    port: int,
    plan: bool,
) -> list[str]:
    if profile not in PROFILES:
        raise SpikeError(f"Perfil desconocido: {profile}")
    command = [
        str(strata_bin),
        "launch",
        "--family",
        DEFAULT_MODEL_FAMILY,
        "--model",
        DEFAULT_MODEL,
        "--data-dir",
        str(root / "models"),
        "--context",
        DEFAULT_CONTEXT,
        "--vision",
        "off",
        "--parallel",
        "1",
        "--gpu-roles",
        PROFILES[profile],
        "--split-engine",
        "off",
        "--tune",
        "skip",
        "--access",
        "local",
        "--port",
        str(port),
        "--api-key-env",
        "STRATA_SPIKE_API_KEY",
        "--no-tui",
        "--output",
        "json",
    ]
    if plan:
        command.append("--plan")
    else:
        command.extend(["--yes", "--restart"])
    return command


def _json_request(url: str, api_key: str) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        raise SpikeError(f"No se pudo leer {url}: {exc}") from exc


def _wait_for_health(base_url: str, api_key: str, timeout_s: int = 180) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last_error = "sin respuesta"
    while time.monotonic() < deadline:
        try:
            health = _json_request(f"{base_url}/health", api_key)
            if health.get("status") == "ok" and health.get("loaded") is True:
                return health
            last_error = f"health={health}"
        except SpikeError as exc:
            last_error = str(exc)
        time.sleep(2)
    raise SpikeError(f"Strata no quedo listo en {timeout_s}s: {last_error}")


def _read_sse(response: BinaryIO, started: float) -> tuple[float, str, int]:
    first_token_at = 0.0
    text_parts: list[str] = []
    completion_tokens = 0
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except ValueError:
            continue
        choices = event.get("choices") or []
        delta = choices[0].get("delta") if choices else None
        content = delta.get("content") if isinstance(delta, dict) else None
        reasoning = delta.get("reasoning_content") if isinstance(delta, dict) else None
        if content or reasoning:
            if not first_token_at:
                first_token_at = time.monotonic()
        if content:
            text_parts.append(str(content))
        usage = event.get("usage")
        if isinstance(usage, dict):
            completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
    if not first_token_at:
        first_token_at = time.monotonic()
    return (first_token_at - started, "".join(text_parts), completion_tokens)


def generate(
    base_url: str, api_key: str, model: str, prompt: dict[str, object]
) -> GenerationResult:
    body = {
        "model": model,
        "messages": prompt["messages"],
        "max_tokens": prompt["max_tokens"],
        "temperature": 0,
        "reasoning_effort": "none",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            ttft_s, text, completion_tokens = _read_sse(response, started)
    except (OSError, urllib.error.HTTPError) as exc:
        raise SpikeError(f"La generacion {prompt['name']} fallo: {exc}") from exc
    total_s = time.monotonic() - started
    rate = completion_tokens / max(total_s - ttft_s, 0.001)
    return GenerationResult(
        prompt=str(prompt["name"]),
        ttft_ms=round(ttft_s * 1000, 2),
        total_ms=round(total_s * 1000, 2),
        completion_tokens=completion_tokens,
        output_tokens_per_second=round(rate, 2),
        text_preview=text[:160],
    )


def check_tool_call(base_url: str, api_key: str, model: str) -> ToolResult:
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Usa la herramienta sumar para calcular 19 + 23."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "sumar",
                    "description": "Suma dos enteros.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "sumar"}},
        "max_tokens": 128,
        "temperature": 0,
        "reasoning_effort": "none",
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            payload = json.loads(response.read())
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        raise SpikeError(f"El smoke de tools fallo: {exc}") from exc
    choices = payload.get("choices") or []
    message = choices[0].get("message") if choices else {}
    calls = message.get("tool_calls") if isinstance(message, dict) else []
    call = calls[0] if calls else {}
    function = call.get("function") if isinstance(call, dict) else {}
    raw_arguments = function.get("arguments") if isinstance(function, dict) else None
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except ValueError:
        arguments = raw_arguments
    usage = payload.get("usage") or {}
    result = ToolResult(
        called=bool(calls),
        name=function.get("name") if isinstance(function, dict) else None,
        arguments=arguments,
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )
    if not result.called or result.name != "sumar":
        raise SpikeError(f"Strata no devolvio la tool forzada esperada: {result}")
    return result


def _write_result(results_root: Path, profile: str, payload: dict[str, object]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = results_root / f"{stamp}_{profile}"
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "run.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def execute_profile(
    strata_bin: Path,
    root: Path,
    results_root: Path,
    *,
    profile: str,
    port: int,
    allow_busy_gpus: bool,
) -> Path:
    preflight = collect_preflight(root, strata_bin, port)
    if preflight["errors"]:
        raise SpikeError("Preflight fallido: " + "; ".join(preflight["errors"]))
    if preflight["gpu_processes"] and not allow_busy_gpus:
        raise SpikeError(
            "Hay procesos de computo en las GPU. Descarga los modelos de oCabra y repite, "
            "o usa --allow-busy-gpus si la coexistencia es deliberada."
        )

    api_key = secrets.token_urlsafe(32)
    environment = _strata_environment(root, api_key)
    command = build_launch_command(strata_bin, root, profile=profile, port=port, plan=False)
    started = time.monotonic()
    launch_output = ""
    error: str | None = None
    health: dict[str, object] = {}
    status: dict[str, object] = {}
    generations: list[GenerationResult] = []
    tool_result: ToolResult | None = None
    gpu_loaded: list[dict[str, object]] = []
    load_time_s = 0.0
    try:
        launch_output = _run(command, timeout=1800, env=environment)
        base_url = f"http://127.0.0.1:{port}"
        health = _wait_for_health(base_url, api_key)
        load_time_s = time.monotonic() - started
        gpu_loaded = _gpu_usage()
        model = str(health.get("model") or "strata")
        for prompt in PROMPTS:
            generations.append(generate(base_url, api_key, model, prompt))
        tool_result = check_tool_call(base_url, api_key, model)
        status = _json_request(f"{base_url}/v1/status", api_key)
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        stop_error = None
        try:
            _run(
                [
                    str(strata_bin),
                    "launch",
                    "--data-dir",
                    str(root / "models"),
                    "--stop",
                    "--output",
                    "json",
                ],
                timeout=180,
                env=environment,
            )
        except SpikeError as exc:
            stop_error = str(exc)
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": "failed" if error else "completed",
            "error": error,
            "stop_error": stop_error,
            "profile": profile,
            "gpu_roles": PROFILES[profile],
            "model": {"family": DEFAULT_MODEL_FAMILY, "name": DEFAULT_MODEL},
            "context": DEFAULT_CONTEXT,
            "parallel": 1,
            "load_time_s": round(load_time_s, 2),
            "elapsed_s": round(time.monotonic() - started, 2),
            "command": [part for part in command if part != api_key],
            "launch_output": launch_output.replace(api_key, "[REDACTED]"),
            "health": health,
            "strata_status": status,
            "gpu_loaded": gpu_loaded,
            "gpu_after_stop": _gpu_usage(),
            "generations": [asdict(result) for result in generations],
            "tool_call": asdict(tool_result) if tool_result else None,
            "preflight": preflight,
            "ocabra": _git_revision(),
        }
        result_path = _write_result(results_root, profile, payload)
        print(result_path)
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--strata-bin")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    plan = subparsers.add_parser("plan")
    plan.add_argument("--profile", choices=sorted(PROFILES), default="3090")
    run = subparsers.add_parser("run")
    run.add_argument("--profile", choices=sorted(PROFILES), default="3090")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--allow-busy-gpus", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    strata_bin = _strata_bin(root, args.strata_bin)
    if args.command == "preflight":
        report = collect_preflight(root, strata_bin, args.port)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ready"] else 2
    if args.command == "plan":
        command = build_launch_command(
            strata_bin, root, profile=args.profile, port=args.port, plan=True
        )
        environment = _strata_environment(root, "plan-only-not-a-secret")
        print(_run(command, timeout=300, env=environment))
        return 0
    if not args.execute:
        raise SpikeError(
            "run requiere --execute porque puede descargar decenas de GB y usar ambas GPU"
        )
    execute_profile(
        strata_bin,
        root,
        args.results_root.expanduser().resolve(),
        profile=args.profile,
        port=args.port,
        allow_busy_gpus=args.allow_busy_gpus,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpikeError as exc:
        print(f"strata-spike: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
