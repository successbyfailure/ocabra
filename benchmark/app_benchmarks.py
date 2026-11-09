#!/usr/bin/env python3
"""Run application-owned benchmark datasets and archive reproducible metadata.

The datasets and scoring code stay in each application's repository.  This
module only checks out (or uses) that repository, invokes a declared suite and
stores the raw result alongside enough context to understand a later manual
comparison.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent
DEFAULT_MANIFEST_DIR = BENCHMARK_DIR / "apps"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results"
MANIFEST_SCHEMA_VERSION = 1


class BenchmarkError(RuntimeError):
    """A benchmark could not be prepared or executed safely."""


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    description: str
    script: Path
    result_glob: str
    models_flag: str | None = None
    prepare_script: Path | None = None


@dataclass(frozen=True)
class AppConfig:
    app: str
    description: str
    repository: str
    default_revision: str
    dataset_root: Path
    suites: dict[str, SuiteConfig]
    manifest_path: Path


@dataclass(frozen=True)
class SourceCheckout:
    root: Path
    requested_revision: str | None
    commit: str | None
    dirty: bool | None
    repository: str | None


def _safe_relative_path(value: object, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{field_name} debe ser una ruta relativa no vacia")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise BenchmarkError(f"{field_name} debe quedar dentro del repositorio: {value}")
    return path


def load_app_config(path: Path) -> AppConfig:
    """Load and validate one application benchmark manifest."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"No se pudo leer el manifiesto {path}: {exc}") from exc

    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise BenchmarkError(f"Version de manifiesto no soportada: {raw.get('schema_version')!r}")

    required_strings = ("app", "description", "repository", "default_revision")
    for field in required_strings:
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise BenchmarkError(f"El manifiesto necesita un campo {field!r} no vacio")

    dataset_root = _safe_relative_path(raw.get("dataset_root"), "dataset_root")
    raw_suites = raw.get("suites")
    if not isinstance(raw_suites, dict) or not raw_suites:
        raise BenchmarkError("El manifiesto debe declarar al menos una suite")

    suites: dict[str, SuiteConfig] = {}
    for name, value in raw_suites.items():
        if not isinstance(name, str) or not name or not isinstance(value, dict):
            raise BenchmarkError("Cada suite debe tener un nombre y configuracion validos")
        script = _safe_relative_path(value.get("script"), f"suites.{name}.script")
        result_glob = value.get("result_glob")
        if not isinstance(result_glob, str) or not result_glob.strip():
            raise BenchmarkError(f"suites.{name}.result_glob no puede estar vacio")
        if Path(result_glob).is_absolute() or ".." in Path(result_glob).parts:
            raise BenchmarkError(f"suites.{name}.result_glob sale del dataset")
        prepare_value = value.get("prepare_script")
        prepare_script = (
            _safe_relative_path(prepare_value, f"suites.{name}.prepare_script")
            if prepare_value is not None
            else None
        )
        models_flag = value.get("models_flag")
        if models_flag is not None and (
            not isinstance(models_flag, str) or not models_flag.startswith("--")
        ):
            raise BenchmarkError(f"suites.{name}.models_flag debe ser una opcion --larga")
        suites[name] = SuiteConfig(
            name=name,
            description=str(value.get("description", "")),
            script=script,
            result_glob=result_glob,
            models_flag=models_flag,
            prepare_script=prepare_script,
        )

    return AppConfig(
        app=raw["app"],
        description=raw["description"],
        repository=raw["repository"],
        default_revision=raw["default_revision"],
        dataset_root=dataset_root,
        suites=suites,
        manifest_path=path,
    )


def discover_apps(manifest_dir: Path = DEFAULT_MANIFEST_DIR) -> dict[str, AppConfig]:
    """Return all valid application manifests keyed by application name."""
    apps: dict[str, AppConfig] = {}
    for path in sorted(manifest_dir.glob("*.json")):
        config = load_app_config(path)
        if config.app in apps:
            raise BenchmarkError(f"Aplicacion duplicada en manifiestos: {config.app}")
        apps[config.app] = config
    return apps


def _run_git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _git_metadata(root: Path) -> tuple[str | None, bool | None, str | None]:
    commit = _run_git(root, "rev-parse", "HEAD")
    if commit is None:
        return None, None, None
    status = _run_git(root, "status", "--porcelain")
    repository = _run_git(root, "remote", "get-url", "origin")
    return commit, bool(status) if status is not None else None, repository


@contextlib.contextmanager
def acquire_source(
    config: AppConfig,
    source: Path | None,
    revision: str | None,
) -> Iterator[SourceCheckout]:
    """Use a local checkout or clone a disposable checkout at a requested revision."""
    if source is not None:
        root = source.resolve()
        if not root.is_dir():
            raise BenchmarkError(f"El checkout local no existe: {root}")
        commit, dirty, repository = _git_metadata(root)
        if revision is not None:
            resolved_revision = _run_git(root, "rev-parse", revision)
            if resolved_revision is None:
                raise BenchmarkError(
                    f"La revisión {revision!r} no existe en el checkout local {root}"
                )
            if commit != resolved_revision:
                raise BenchmarkError(
                    "El checkout local no está en la revisión solicitada: "
                    f"HEAD={commit or 'sin Git'}, revisión={resolved_revision}"
                )
        yield SourceCheckout(root, revision, commit, dirty, repository)
        return

    requested_revision = revision or config.default_revision
    with tempfile.TemporaryDirectory(prefix=f"ocabra-{config.app}-") as temp_dir:
        root = Path(temp_dir) / config.app
        clone = subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", config.repository, str(root)],
            text=True,
            capture_output=True,
        )
        if clone.returncode != 0:
            raise BenchmarkError(f"No se pudo clonar {config.repository}: {clone.stderr.strip()}")
        checkout = subprocess.run(
            ["git", "-C", str(root), "checkout", "--quiet", "--detach", requested_revision],
            text=True,
            capture_output=True,
        )
        if checkout.returncode != 0:
            raise BenchmarkError(
                f"No se pudo fijar la revision {requested_revision!r}: {checkout.stderr.strip()}"
            )
        commit, dirty, repository = _git_metadata(root)
        yield SourceCheckout(root, requested_revision, commit, dirty, repository)


def _snapshot_results(dataset_root: Path, result_glob: str) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for path in dataset_root.glob(result_glob):
        if path.is_file():
            stat = path.stat()
            snapshot[path.resolve()] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _changed_result(
    dataset_root: Path,
    result_glob: str,
    before: dict[Path, tuple[int, int]],
) -> Path | None:
    changed: list[Path] = []
    for path in dataset_root.glob(result_glob):
        if not path.is_file():
            continue
        resolved = path.resolve()
        stat = path.stat()
        if before.get(resolved) != (stat.st_mtime_ns, stat.st_size):
            changed.append(path)
    return max(changed, key=lambda item: item.stat().st_mtime_ns) if changed else None


def _directory_digest(root: Path) -> str:
    """Hash versioned dataset inputs while excluding generated and downloaded files."""
    digest = hashlib.sha256()
    ignored_parts = {"__pycache__"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ignored_parts.intersection(path.parts):
            continue
        relative = path.relative_to(root)
        if any(part.startswith("results_") for part in relative.parts):
            continue
        if path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}:
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _server_version(base_url: str, api_key: str) -> dict[str, object]:
    url = f"{base_url.rstrip('/')}/version"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
            return {"url": base_url, "version": body.get("version"), "reachable": True}
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        return {"url": base_url, "version": None, "reachable": False, "error": str(exc)}


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stream_command(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def _run_id(app: str, suite: str, commit: str | None) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    revision = commit[:8] if commit else "local"
    return f"{stamp}_{app}_{suite}_{revision}"


def execute_benchmark(
    config: AppConfig,
    suite_name: str,
    *,
    source: Path | None,
    revision: str | None,
    base_url: str,
    api_key: str,
    models: str | None,
    prepare: bool,
    suite_args: list[str],
    output_dir: Path,
    dry_run: bool = False,
) -> tuple[int, Path | None]:
    """Execute one suite and return its process code and archived run directory."""
    if suite_name not in config.suites:
        raise BenchmarkError(f"Suite desconocida para {config.app}: {suite_name}")
    suite = config.suites[suite_name]

    with acquire_source(config, source, revision) as checkout:
        dataset_root = checkout.root / config.dataset_root
        script = dataset_root / suite.script
        if not dataset_root.is_dir():
            raise BenchmarkError(f"No existe el dataset esperado: {dataset_root}")
        if not script.is_file():
            raise BenchmarkError(f"No existe el runner de la suite: {script}")

        command = [sys.executable, "-I", str(script)]
        if models:
            if not suite.models_flag:
                raise BenchmarkError(f"La suite {suite_name} no admite --models")
            command.extend([suite.models_flag, models])
        command.extend(suite_args)

        prepare_command: list[str] | None = None
        if prepare:
            if suite.prepare_script is None:
                raise BenchmarkError(f"La suite {suite_name} no necesita preparacion")
            prepare_path = dataset_root / suite.prepare_script
            if not prepare_path.is_file():
                raise BenchmarkError(f"No existe el script de preparacion: {prepare_path}")
            prepare_command = [sys.executable, "-I", str(prepare_path)]

        if dry_run:
            print(f"Fuente: {checkout.root}")
            print(f"Revision: {checkout.commit or 'sin Git'}")
            if prepare_command:
                print("Preparacion:", " ".join(prepare_command))
            print("Benchmark:", " ".join(command))
            return 0, None

        run_dir = (
            output_dir.resolve()
            / config.app
            / suite_name
            / _run_id(config.app, suite_name, checkout.commit)
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        started = datetime.now(UTC)
        before = _snapshot_results(dataset_root, suite.result_glob)
        environment = dict(os.environ)
        environment["OPENAI_COMPATIBLE_URL"] = base_url
        environment["OPENAI_COMPATIBLE_KEY"] = api_key

        exit_code = 0
        if prepare_command:
            exit_code = _stream_command(
                prepare_command, checkout.root, environment, run_dir / "prepare.log"
            )
        if exit_code == 0:
            exit_code = _stream_command(command, checkout.root, environment, run_dir / "runner.log")

        result_path = _changed_result(dataset_root, suite.result_glob, before)
        archived_result: str | None = None
        if result_path is not None:
            destination = run_dir / "results.json"
            shutil.copy2(result_path, destination)
            archived_result = destination.name

        finished = datetime.now(UTC)
        ocabra_commit, ocabra_dirty, _ = _git_metadata(PROJECT_ROOT)
        metadata = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "status": "completed" if exit_code == 0 and result_path else "failed",
            "exit_code": exit_code,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": round((finished - started).total_seconds(), 3),
            "application": config.app,
            "suite": suite_name,
            "models": models.split(",") if models else None,
            "suite_args": suite_args,
            "dataset": {
                "repository": checkout.repository or config.repository,
                "requested_revision": checkout.requested_revision,
                "commit": checkout.commit,
                "dirty": checkout.dirty,
                "root": config.dataset_root.as_posix(),
                "sha256": _directory_digest(dataset_root),
            },
            "ocabra_client": {"commit": ocabra_commit, "dirty": ocabra_dirty},
            "server": _server_version(base_url, api_key),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "command": [Path(command[0]).name, *command[1:]],
            "result_file": archived_result,
        }
        _write_json(run_dir / "run.json", metadata)
        if result_path is None and exit_code == 0:
            print(f"La suite termino sin generar un resultado que coincida con {suite.result_glob}")
            return 2, run_dir
        return exit_code, run_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta manualmente datasets de aplicaciones contra oCabra"
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR, help=argparse.SUPPRESS
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="Lista aplicaciones y suites disponibles")

    run_parser = subparsers.add_parser("run", help="Ejecuta y archiva una suite")
    run_parser.add_argument("app")
    run_parser.add_argument("suite")
    run_parser.add_argument("--source", type=Path, help="Checkout local de la aplicacion")
    run_parser.add_argument("--revision", help="Revision Git; por defecto la del manifiesto")
    run_parser.add_argument("--base-url", help="URL de oCabra; o OPENAI_COMPATIBLE_URL")
    run_parser.add_argument(
        "--api-key-env",
        default="OPENAI_COMPATIBLE_KEY",
        help="Variable que contiene la clave (nunca se guarda)",
    )
    run_parser.add_argument("--models", help="Lista de modelos separada por comas")
    run_parser.add_argument("--prepare", action="store_true", help="Prepara assets de la suite")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument(
        "--suite-arg",
        dest="suite_args",
        action="append",
        default=[],
        help="Argumento adicional del runner; se puede repetir (usa = para valores --)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        apps = discover_apps(args.manifest_dir)
        if args.command == "list":
            for app in apps.values():
                print(f"{app.app}: {app.description}")
                for suite in app.suites.values():
                    print(f"  {suite.name}: {suite.description}")
            return 0

        config = apps.get(args.app)
        if config is None:
            raise BenchmarkError(f"Aplicacion desconocida: {args.app}")
        base_url = args.base_url or os.environ.get("OPENAI_COMPATIBLE_URL")
        if not base_url:
            raise BenchmarkError(
                "Indica --base-url o OPENAI_COMPATIBLE_URL; no se elige un servidor implicitamente"
            )
        exit_code, run_dir = execute_benchmark(
            config,
            args.suite,
            source=args.source,
            revision=args.revision,
            base_url=base_url,
            api_key=os.environ.get(args.api_key_env, ""),
            models=args.models,
            prepare=args.prepare,
            suite_args=args.suite_args,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        if run_dir:
            print(f"Ejecucion archivada en {run_dir}")
        return exit_code
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
