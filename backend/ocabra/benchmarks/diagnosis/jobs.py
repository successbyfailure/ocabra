"""Diagnósticos como trabajos en disco, ejecutados por la cola de benchmarks.

Un diagnóstico vive en ``<benchmark_dir>/diagnoses/<id>/``:

- ``job.json``: petición, estado (``queued``/``running``/``completed``/
  ``failed``), fase y progreso.
- ``report.json`` / ``report.md``: el informe.
- ``replays.jsonl``: cada repetición (caso, ajuste, resultado).
- ``candidates.train.jsonl`` / ``candidates.eval.jsonl``: casos «modelo».

Lo ejecuta el worker de :class:`BenchmarkManager` cuando no hay ningún
benchmark en cola: las repeticiones generan tráfico y no deben coincidir con
una medida. Entran por la cola normal de oCabra con la identidad de servicio de
los benchmarks, igual que los runners.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import sys
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog

from ocabra.benchmarks.diagnosis import cases as case_module
from ocabra.benchmarks.diagnosis import serving
from ocabra.benchmarks.diagnosis.classify import classify_case, summarize
from ocabra.benchmarks.diagnosis.reference import (
    Reference,
    load_reference,
    resolve_reference_repo,
)
from ocabra.benchmarks.diagnosis.report import render_markdown, write_candidates
from ocabra.benchmarks.models import bench_profile
from ocabra.config import settings

logger = structlog.get_logger(__name__)

TERMINAL = frozenset({"completed", "failed", "cancelled"})
SCRIPTS = Path(__file__).resolve().parent / "scripts"
_IFEVAL_DEPS = ("absl-py", "langdetect", "nltk", "immutabledict")
# Apps que traen su propio script de repetición (contrato ocabra_replay.py):
# repiten el caso por su stack real; el muestreo a comparar lo pone el servidor.
EXTERNAL_REPLAY = {
    "corrala": "backend/scripts/llm_benchmark/ocabra_replay.py",
    "escriba": "eval/datasets/llm/ocabra_replay.py",
    "fulcrum": "benchmarks/orchestration/ocabra_replay.py",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DiagnosisStore:
    """Lectura y escritura de los trabajos en disco."""

    def __init__(self, root: Path) -> None:
        self.root = root / "diagnoses"

    def path(self, job_id: str) -> Path:
        uuid.UUID(job_id)  # nunca rutas arbitrarias
        return self.root / job_id

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "status": "queued",
            "phase": "queued",
            "request": request,
            "progress": {"completed": 0, "total": 0},
            "error": None,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
        }
        directory = self.path(job_id)
        directory.mkdir(parents=True, exist_ok=False)
        self.save(job)
        return job

    def save(self, job: dict[str, Any]) -> None:
        directory = self.path(job["id"])
        temporary = directory / "job.json.tmp"
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
        temporary.replace(directory / "job.json")

    def load(self, job_id: str) -> dict[str, Any] | None:
        try:
            return json.loads((self.path(job_id) / "job.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def all(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        jobs = []
        for directory in self.root.iterdir():
            job = self.load(directory.name) if directory.is_dir() else None
            if job:
                jobs.append(job)
        return sorted(jobs, key=lambda item: item["created_at"], reverse=True)

    def next_queued(self) -> dict[str, Any] | None:
        queued = [job for job in self.all() if job["status"] == "queued"]
        return min(queued, key=lambda item: item["created_at"]) if queued else None

    def report(self, job_id: str) -> dict[str, Any] | None:
        try:
            return json.loads((self.path(job_id) / "report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def delete(self, job_id: str) -> bool:
        """Borra un diagnóstico terminado y todos sus ficheros."""
        job = self.load(job_id)
        if job is None:
            return False
        if job["status"] not in TERMINAL:
            raise ValueError("Cancela el diagnóstico antes de borrarlo")
        shutil.rmtree(self.path(job_id), ignore_errors=True)
        return True

    def cancel_queued(self, job_id: str) -> dict[str, Any] | None:
        job = self.load(job_id)
        if job is not None and job["status"] == "queued":
            job.update(status="cancelled", phase="cancelled", error="cancelado", finished_at=_now())
            self.save(job)
        return job

    def purge(self, retention_days: int, now: datetime | None = None) -> int:
        """Borra los diagnósticos terminados hace más de ``retention_days`` días."""
        if retention_days <= 0:
            return 0
        now = now or datetime.now(UTC)
        removed = 0
        for job in self.all():
            finished = job.get("finished_at") or job.get("created_at")
            if job["status"] not in TERMINAL or not finished:
                continue
            age = now - datetime.fromisoformat(finished)
            if age.total_seconds() > retention_days * 86400:
                shutil.rmtree(self.path(job["id"]), ignore_errors=True)
                removed += 1
        return removed

    def recover_interrupted(self) -> None:
        for job in self.all():
            if job["status"] == "running":
                job["status"] = "failed"
                job["phase"] = "failed"
                job["error"] = "interrumpido por un reinicio del api; vuelve a lanzarlo"
                job["finished_at"] = _now()
                self.save(job)


class DiagnosisCancelled(Exception):
    """Se pidió cancelar el diagnóstico en curso."""


class DiagnosisRunner:
    """Ejecuta un diagnóstico de principio a fin."""

    def __init__(self, app: Any, store: DiagnosisStore, api_key: str) -> None:
        self._app = app
        self._store = store
        self._api_key = api_key
        self._cancelled = False
        self._process: asyncio.subprocess.Process | None = None
        self.job_id: str | None = None

    def cancel(self) -> None:
        """Corta el diagnóstico: mata el script en curso y para en la siguiente fase."""
        self._cancelled = True
        if self._process is not None and self._process.returncode is None:
            self._process.kill()

    def _check(self) -> None:
        if self._cancelled:
            raise DiagnosisCancelled()

    # ── Entrada ─────────────────────────────────────────────────────────

    async def run(self, job: dict[str, Any]) -> None:
        self.job_id = job["id"]
        job.update(status="running", phase="cases", started_at=_now(), error=None)
        self._store.save(job)
        try:
            await self._run(job)
            self._check()
        except DiagnosisCancelled:
            job.update(status="cancelled", phase="cancelled", error="cancelado")
        except Exception as exc:  # noqa: BLE001 — el informe dice por qué
            if self._cancelled:  # matar el script hace fallar la fase en curso
                job.update(status="cancelled", phase="cancelled", error="cancelado")
            else:
                logger.exception("benchmark_diagnosis_failed", job_id=job["id"])
                job.update(
                    status="failed", phase="failed", error=f"{type(exc).__name__}: {exc}"[:2000]
                )
        else:
            job.update(status="completed", phase="completed")
        job["finished_at"] = _now()
        self._store.save(job)

    async def _attempt(self, attempt_id: str) -> tuple[Any, Any]:
        from ocabra.database import AsyncSessionLocal
        from ocabra.db.benchmarks import BenchmarkAttempt, BenchmarkRun

        async with AsyncSessionLocal() as session:
            attempt = await session.get(BenchmarkAttempt, uuid.UUID(attempt_id))
            if attempt is None:
                raise LookupError("intento de benchmark no encontrado")
            run = await session.get(BenchmarkRun, attempt.run_id)
            return attempt, run

    async def _run(self, job: dict[str, Any]) -> None:
        request = job["request"]
        attempt, run = await self._attempt(request["attempt_id"])
        if attempt.status != "completed":
            raise ValueError("solo se diagnostican intentos completados")
        directory = self._store.path(job["id"])
        workspace = Path(attempt.workspace_path)
        model = attempt.model
        failed = await asyncio.to_thread(
            case_module.failed_cases, run.app, run.suite, workspace, model
        )
        cases = [case.to_dict() for case in failed]
        if run.app == "bfcl" and cases:
            job["phase"] = "rescore"
            self._store.save(job)
            await self._bfcl_rescore(directory, workspace, cases)
        self._check()
        job["phase"] = "reference"
        job["progress"] = {"completed": 0, "total": 0, "cases": len(cases)}
        self._store.save(job)

        app_state = self._app.state
        profile = await bench_profile(app_state.profile_registry, model)
        if profile is None:
            raise LookupError(f"perfil '{model}' no encontrado")
        state = await app_state.model_manager.get_state(profile.base_model_id)
        if state is None:
            raise LookupError(f"modelo '{profile.base_model_id}' no encontrado")
        model_key = getattr(state, "model_key", None) or model
        reference = await load_reference(
            resolve_reference_repo(
                str(model_key),
                getattr(state, "artifact", None),
                getattr(state, "extra_config", None),
                request.get("reference_repo"),
            )
        )

        findings: list[str] = []
        if reference.error:
            findings.append(f"Sin ficha oficial: {reference.error}")
        worker = None
        if serving.is_llama_server(getattr(state, "backend_type", None)):
            worker = await serving.find_loaded_worker(
                app_state.model_manager, profile.base_model_id
            )
            if worker is None and request.get("allow_load"):
                job["phase"] = "load"
                self._store.save(job)
                await self._warm_up(model)
                worker = await serving.find_loaded_worker(
                    app_state.model_manager, profile.base_model_id
                )
        if worker is None:
            findings.append(
                "El modelo no está cargado: sin plantilla viva ni parámetros por defecto del "
                "servidor" + ("" if request.get("allow_load") else ", y sin repeticiones")
            )

        async with httpx.AsyncClient() as client:
            probe = serving.WorkerProbe(worker, client) if worker else None
            params, context_per_slot = await self._params(
                app_state, model, cases, probe, reference, findings
            )
            template = await self._template_probes(probe, reference, params, findings)
            sample = self._sample(cases, int(request.get("max_cases") or 30))
            per_case_template, prompt_tokens = await self._case_templates(
                probe, reference, params, cases, sample
            )

            replays: dict[str, dict[str, list[dict[str, Any]]]] = {}
            can_replay = worker is not None or bool(request.get("allow_load"))
            replayable = [
                case for case in cases if case["case_id"] in sample and case["replayable"]
            ]
            self._check()
            if can_replay and replayable:
                job["phase"] = "replay"
                self._store.save(job)
                replay = self._replay_external if run.app in EXTERNAL_REPLAY else self._replay
                replays = await replay(
                    job, directory, workspace, run.app, model, replayable, reference, request
                )

        verdicts = []
        for case in cases:
            category, evidence = classify_case(
                case,
                template=per_case_template.get(case["case_id"]),
                sampling_diffs=params.get("differences") or {},
                replays=replays.get(case["case_id"], {}),
                context_per_slot=context_per_slot,
                prompt_tokens=prompt_tokens.get(case["case_id"]),
            )
            if case["case_id"] not in sample and category == "sin_verificar" and case["replayable"]:
                evidence = ["fuera de la muestra repetida (sube max_cases)"]
            verdicts.append(
                {
                    "case_id": case["case_id"],
                    "category": category,
                    "case_category": case["category"],
                    "failure": case["failure"],
                    "evidence": evidence,
                    "template": per_case_template.get(case["case_id"]),
                    "replays": replays.get(case["case_id"], {}),
                }
            )

        harness = sum(
            1 for case in cases if (case.get("signals") or {}).get("harness_dot_to_underscore")
        )
        if harness:
            findings.append(
                f"BFCL: {harness} fallos son el arnés (punto→guion bajo sin underscore_to_dot en "
                "el runner de oCabra); la nota de BFCL de este modelo está infravalorada"
            )
        source = {"run_id": str(run.id), "attempt_id": str(attempt.id), "revision": run.revision}
        by_id = {item["case_id"]: item for item in verdicts}
        candidates = await asyncio.to_thread(
            write_candidates, directory, model, cases, by_id, replays, source
        )
        report = {
            "id": job["id"],
            "model": model,
            "app": run.app,
            "suite": run.suite,
            "run_id": str(run.id),
            "attempt_id": str(attempt.id),
            "repeats": int(request.get("repeats") or 3),
            "replayed_cases": len(replays),
            "reference": reference.to_dict(),
            "params": params,
            "template": template,
            "findings": findings,
            "summary": summarize(verdicts),
            "cases": verdicts,
            "candidates": candidates,
            "created_at": _now(),
        }
        (directory / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (directory / "report.md").write_text(render_markdown(report), encoding="utf-8")

    async def _warm_up(self, model: str) -> None:
        """Carga el modelo por la cola normal con una petición mínima."""
        async with httpx.AsyncClient(timeout=1800) as client:
            response = await client.post(
                f"{settings.benchmark_base_url.rstrip('/')}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Di hola."}],
                    "max_tokens": 1,
                },
            )
            response.raise_for_status()

    # ── Parámetros ──────────────────────────────────────────────────────

    async def _params(
        self,
        app_state: Any,
        model: str,
        cases: list[dict[str, Any]],
        probe: serving.WorkerProbe | None,
        reference: Reference,
        findings: list[str],
    ) -> tuple[dict[str, Any], int]:
        sample = next((case for case in cases if case.get("messages")), cases[0] if cases else None)
        client_body = dict(sample["client_body"]) if sample else {}
        body, _profile, state = await serving.backend_body(
            app_state,
            model,
            client_body,
            sample.get("messages") if sample else None,
            sample.get("tools") if sample else None,
        )
        defaults = await probe.server_defaults() if probe else {}
        effective = serving.effective_params(defaults, body)
        differences = serving.sampling_differences(effective, reference.sampling)
        if reference.suppress_tokens:
            findings.append(
                f"La ficha oficial suprime los tokens {reference.suppress_tokens} y llama.cpp no "
                "lo hace por defecto (habría que mandarlos como logit_bias)"
            )
        context_per_slot = 0
        if probe:
            context_per_slot = (await probe.context())["n_ctx_slot"]
        if not context_per_slot:
            capabilities = getattr(state, "capabilities", None)
            context_per_slot = int(getattr(capabilities, "context_per_request", 0) or 0)
        return (
            {
                "client_body": client_body,
                "server_defaults": defaults,
                "effective": effective,
                "recommended": reference.sampling,
                "differences": differences,
                "context_per_slot": context_per_slot,
            },
            context_per_slot,
        )

    # ── Plantilla ───────────────────────────────────────────────────────

    async def _official(self, reference: Reference, messages, tools, kwargs) -> str:
        assert reference.chat_template is not None
        return await serving.render_official(
            reference.chat_template,
            messages,
            tools=tools,
            bos_token=reference.bos_token,
            eos_token=reference.eos_token,
            kwargs=kwargs,
        )

    async def _template_probes(
        self,
        probe: serving.WorkerProbe | None,
        reference: Reference,
        params: dict[str, Any],
        findings: list[str],
    ) -> dict[str, Any]:
        if probe is None:
            return {}
        props = await probe.props()
        caps = props.get("chat_template_caps") or {}
        live_bos = props.get("bos_token")
        kwargs = dict((params.get("effective") or {}).get("chat_template_kwargs") or {})
        probes: dict[str, dict[str, Any]] = {}
        user = [{"role": "user", "content": "Hola, ¿qué tal?"}]
        if "enable_thinking" not in kwargs:
            # Sin fijarlo, cada lado usa su valor por defecto (el servidor suele
            # pensar y la plantilla oficial no): se compara con el del servidor,
            # que es lo que de verdad ve el modelo.
            default = await probe.apply_template(user, None, kwargs)
            thinking = await probe.apply_template(user, None, {**kwargs, "enable_thinking": True})
            kwargs["enable_thinking"] = default == thinking
        params["template_kwargs"] = kwargs

        # reasoning_effort: ¿cambia de verdad el prompt?
        low = await probe.apply_template(user, None, {**kwargs, "reasoning_effort": "low"})
        high = await probe.apply_template(user, None, {**kwargs, "reasoning_effort": "high"})
        decorative = low == high
        probes["reasoning_effort"] = {
            "verdict": "decorativo" if decorative else "cambia el prompt",
            "detail": f"la plantilla declara supports_reasoning_effort={caps.get('supports_reasoning_effort')}",
        }
        if decorative:
            findings.append(
                "reasoning_effort no cambia el prompt de esta plantilla (es decorativo)"
            )
        on = await probe.apply_template(user, None, {**kwargs, "enable_thinking": True})
        off = await probe.apply_template(user, None, {**kwargs, "enable_thinking": False})
        probes["enable_thinking"] = {
            "verdict": "sin efecto" if on == off else "cambia el prompt",
            "detail": "enable_thinking=true frente a false",
        }
        as_list = await probe.apply_template(
            [{"role": "user", "content": [{"type": "text", "text": "Hola, ¿qué tal?"}]}],
            None,
            kwargs,
        )
        base = await probe.apply_template(user, None, kwargs)
        probes["content_lista"] = {
            "verdict": "igual" if as_list == base else "DISTINTO",
            "detail": "content como lista de partes frente a string",
        }
        if as_list != base:
            findings.append("content como lista se renderiza distinto que como string")
        system = await probe.apply_template(
            [{"role": "system", "content": "MARCA-SISTEMA"}, *user], None, kwargs
        )
        probes["rol_system"] = {
            "verdict": "presente" if "MARCA-SISTEMA" in system else "PERDIDO",
            "detail": "el texto del rol system aparece en el prompt",
        }
        if "MARCA-SISTEMA" not in system:
            findings.append("el rol system se pierde al renderizar")
        if live_bos:
            tokens = await probe.tokenize(base, add_special=True)
            bos_ids = await probe.tokenize(live_bos, add_special=False)
            double = len(bos_ids) == 1 and tokens[:2] == bos_ids * 2
            probes["bos"] = {
                "verdict": "DUPLICADO" if double else "correcto",
                "detail": f"primeros tokens {tokens[:3]}, BOS={bos_ids}",
            }
            if double:
                findings.append(
                    "BOS duplicado: la plantilla lo pone y el tokenizador lo añade otra vez"
                )
        if reference.chat_template:
            conversation = [
                {"role": "system", "content": "Eres un asistente."},
                {"role": "user", "content": "¿Qué tiempo hace en Madrid?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "Madrid"}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 21}'},
                {"role": "user", "content": "¿Y mañana?"},
            ]
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Tiempo en una ciudad",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
            live_messages = json.loads(json.dumps(conversation))
            # llama-server exige los argumentos como string JSON.
            live_messages[2]["tool_calls"][0]["function"]["arguments"] = '{"city": "Madrid"}'
            try:
                live = await probe.apply_template(live_messages, tools, kwargs)
                official = await self._official(reference, conversation, tools, kwargs)
                comparison = serving.compare_prompts(
                    live, official, live_bos=live_bos, official_bos=reference.bos_token
                )
            except Exception as exc:  # noqa: BLE001
                comparison = {"status": "error", "diff": [str(exc)[:300]]}
            probes["conversacion_con_tools"] = {
                "verdict": comparison["status"],
                "detail": "system + tool_call + resultado + nuevo turno frente a la plantilla oficial",
                "diff": comparison["diff"],
                "first_difference": comparison.get("first_difference"),
            }
            if comparison["status"] == "different":
                findings.append(
                    "La plantilla servida no produce el mismo prompt que la oficial en una "
                    "conversación con tools (ver diff)"
                )
        return {"caps": caps, "probes": probes}

    def _sample(self, cases: list[dict[str, Any]], limit: int) -> set[str]:
        """Muestra estratificada por categoría de los casos que hay que repetir."""
        pending = [
            case
            for case in cases
            if not (case.get("signals") or {}).get("harness_dot_to_underscore")
        ]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in pending:
            groups[case["category"]].append(case)
        rng = random.Random(0)
        for items in groups.values():
            rng.shuffle(items)
        chosen: list[str] = []
        while len(chosen) < limit and any(groups.values()):
            for name in sorted(groups):
                if groups[name] and len(chosen) < limit:
                    chosen.append(groups[name].pop()["case_id"])
        return set(chosen)

    async def _case_templates(
        self,
        probe: serving.WorkerProbe | None,
        reference: Reference,
        params: dict[str, Any],
        cases: list[dict[str, Any]],
        sample: set[str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        templates: dict[str, dict[str, Any]] = {}
        tokens: dict[str, int] = {}
        if probe is None:
            return templates, tokens
        props = await probe.props()
        kwargs = dict(
            params.get("template_kwargs")
            or (params.get("effective") or {}).get("chat_template_kwargs")
            or {}
        )
        for case in cases:
            if case["case_id"] not in sample or not case.get("messages"):
                continue
            try:
                live = await probe.apply_template(case["messages"], case.get("tools"), kwargs)
                tokens[case["case_id"]] = len(await probe.tokenize(live, add_special=True))
                if reference.chat_template:
                    official = await self._official(
                        reference, case["messages"], case.get("tools"), kwargs
                    )
                    templates[case["case_id"]] = serving.compare_prompts(
                        live,
                        official,
                        live_bos=props.get("bos_token"),
                        official_bos=reference.bos_token,
                    )
            except Exception as exc:  # noqa: BLE001 — se anota en el caso
                templates[case["case_id"]] = {"status": "error", "diff": [str(exc)[:300]]}
        return templates, tokens

    # ── Repeticiones ────────────────────────────────────────────────────

    def _replay_command(self, app: str, workspace: Path) -> tuple[list[str], Path, Path]:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("hace falta uv para repetir los casos")
        source = workspace / "source"
        if app == "ifeval":
            command = [uv, "run", "--no-project"]
            for dependency in _IFEVAL_DEPS:
                command += ["--with", dependency]
            return [*command, "python", "-I", str(SCRIPTS / "ifeval_replay.py")], source, source
        if app == "bfcl":
            project = source / "berkeley-function-call-leaderboard"
            return (
                [uv, "run", "--project", str(project), "--with", "soundfile", "python", "-I",
                 str(SCRIPTS / "bfcl_replay.py")],
                project,
                project,
            )  # fmt: skip
        raise ValueError(f"sin repetición para {app}")

    async def _run_script(
        self, app: str, workspace: Path, directory: Path, spec: dict[str, Any], name: str
    ) -> Path:
        spec_path = directory / f"{name}_input.json"
        output = directory / f"{name}.jsonl"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        output.write_text("", encoding="utf-8")
        command, cwd, source_root = self._replay_command(app, workspace)
        command += [
            "--input",
            str(spec_path),
            "--output",
            str(output),
            "--source-root",
            str(source_root),
        ]
        with (directory / f"{name}.log").open("ab") as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=self._script_env(),
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            self._process = process
            await asyncio.wait_for(process.wait(), timeout=1800)
            self._process = None
        self._check()
        if process.returncode != 0:
            raise RuntimeError(f"{name} terminó con código {process.returncode}; ver {name}.log")
        return output

    def _script_env(self) -> dict[str, str]:
        root = serving.benchmark_root()
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        env.update(
            {
                "OPENAI_COMPATIBLE_KEY": self._api_key,
                "UV_CACHE_DIR": str(root / "uv" / "cache"),
                "UV_PYTHON_INSTALL_DIR": str(root / "uv" / "python"),
                "NLTK_DATA": str(root / "uv" / "nltk_data"),
                "OCABRA_BFCL_ENV": "1",
            }
        )
        return env

    async def _bfcl_rescore(
        self, directory: Path, workspace: Path, cases: list[dict[str, Any]]
    ) -> None:
        """Re-puntúa con ``underscore_to_dot``: el arnés queda medido, no estimado."""
        spec = {
            "mode": "rescore",
            "cases": [{"case_id": case["case_id"], "payload": case["payload"]} for case in cases],
        }
        output = await self._run_script("bfcl", workspace, directory, spec, "rescore")
        rows = {
            row["case_id"]: row
            for row in (
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line
            )
        }
        for case in cases:
            row = rows.get(case["case_id"]) or {}
            signals = case.setdefault("signals", {})
            if row.get("passed") is True:
                signals["harness_dot_to_underscore"] = True
            elif row.get("passed") is False:
                signals["harness_dot_to_underscore"] = False
                if row.get("error_type") and row["error_type"] not in case["failure"]:
                    case["failure"] = f"{row['error_type']} (re-puntuado con underscore_to_dot)"

    async def _replay_external(
        self,
        job: dict[str, Any],
        directory: Path,
        workspace: Path,
        app: str,
        model: str,
        cases: list[dict[str, Any]],
        reference: Reference,
        request: dict[str, Any],
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Repite con el ``ocabra_replay.py`` de la app.

        Una pasada con el muestreo del benchmark («original») y otra con el
        del fabricante («recomendado»), que el servidor aplica pisando el del
        runner (mismo override que las ejecuciones con muestreo recomendado).
        """
        source = workspace / "source"
        script = source / EXTERNAL_REPLAY[app]
        if not script.is_file():
            raise RuntimeError(
                f"La revisión medida de {app} no trae {EXTERNAL_REPLAY[app]}: vuelve a medir "
                "con la versión actual para poder repetir sus casos"
            )
        repeats = max(1, min(10, int(request.get("repeats") or 3)))
        passes = [("original", model, 1, None), ("recomendado", model, repeats, reference.sampling)]
        if request.get("compare_model"):
            passes.append(("comparado", request["compare_model"], repeats, reference.sampling))
        case_ids = [str(case["payload"]["case_id"]) for case in cases]
        by_replay_id = {str(case["payload"]["case_id"]): case["case_id"] for case in cases}
        cases_path = directory / "replay_cases.json"
        cases_path.write_text(json.dumps({"case_ids": case_ids}), encoding="utf-8")
        total = len(cases) * sum(n for _, _, n, _ in passes)
        job["progress"] = {**job.get("progress", {}), "completed": 0, "total": total}
        self._store.save(job)
        env = self._script_env()
        env["OPENAI_COMPATIBLE_URL"] = settings.benchmark_base_url
        if app == "corrala":
            manager = getattr(self._app.state, "benchmark_manager", None)
            if manager is not None:
                env["CORRALA_BENCH_DATABASE_URL"] = await manager._corrala_database_url()
        results: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        done = 0
        for name, target, n, sampling in passes:
            output = directory / f"replay_{name}.jsonl"
            output.write_text("", encoding="utf-8")
            command = [
                sys.executable, "-I", str(script), "--model", target,
                "--cases", str(cases_path), "--repeats", str(n), "--output", str(output),
            ]  # fmt: skip
            await self._run_with_override(command, source, env, target, sampling, directory, name)
            for line in output.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                case_id = by_replay_id.get(str(row.get("case_id")), str(row.get("case_id")))
                row["setting"] = name
                results[case_id][name].append(row)
            done += len(cases) * n
            job["progress"]["completed"] = done
            self._store.save(job)
        with (directory / "replays.jsonl").open("w", encoding="utf-8") as handle:
            for case_id, by_setting in results.items():
                for rows in by_setting.values():
                    for row in rows:
                        handle.write(
                            json.dumps({**row, "case_id": case_id}, ensure_ascii=False) + "\n"
                        )
        return {key: dict(value) for key, value in results.items()}

    async def _run_with_override(
        self,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        profile: str,
        sampling: dict[str, Any] | None,
        directory: Path,
        name: str,
    ) -> None:
        """Lanza el script; con ``sampling`` activa el override del servidor mientras dura."""
        manager = getattr(self._app.state, "benchmark_manager", None)
        if sampling and manager is not None:
            manager.sampling_override = {
                "profile": profile,
                "params": sampling,
                "api_key": self._api_key,
            }
        try:
            with (directory / f"replay_{name}.log").open("ab") as log:
                process = await asyncio.create_subprocess_exec(
                    *command, cwd=str(cwd), env=env, stdout=log, stderr=asyncio.subprocess.STDOUT
                )
                self._process = process
                await asyncio.wait_for(process.wait(), timeout=4 * 3600)
                self._process = None
        finally:
            if sampling and manager is not None:
                manager.sampling_override = None
        self._check()
        if process.returncode not in (0, None):
            # El contrato sale ≠0 solo si todo falló por infraestructura.
            raise RuntimeError(f"replay {name} terminó con código {process.returncode}")

    async def _replay(
        self,
        job: dict[str, Any],
        directory: Path,
        workspace: Path,
        app: str,
        model: str,
        cases: list[dict[str, Any]],
        reference: Reference,
        request: dict[str, Any],
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        repeats = max(1, min(10, int(request.get("repeats") or 3)))
        client_body = dict(cases[0]["client_body"])
        recommended = dict(reference.sampling)
        # Mismo tope de salida que el benchmark: lo que se mide es el muestreo.
        for key in ("max_tokens", "max_completion_tokens"):
            if key in client_body:
                recommended[key] = client_body[key]
        settings_ = [
            {"name": "original", "model": model, "params": client_body, "repeats": 1},
            {"name": "recomendado", "model": model, "params": recommended, "repeats": repeats},
        ]
        if request.get("compare_model"):
            settings_.append(
                {
                    "name": "comparado",
                    "model": request["compare_model"],
                    "params": recommended,
                    "repeats": repeats,
                }
            )
        spec = {
            "base_url": settings.benchmark_base_url,
            "cases": [{"case_id": case["case_id"], "payload": case["payload"]} for case in cases],
            "settings": settings_,
        }
        spec_path = directory / "replay_input.json"
        output = directory / "replays.jsonl"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        output.write_text("", encoding="utf-8")
        total = len(cases) * sum(item["repeats"] for item in settings_)
        job["progress"] = {**job.get("progress", {}), "completed": 0, "total": total}
        self._store.save(job)

        command, cwd, source_root = self._replay_command(app, workspace)
        command += [
            "--input",
            str(spec_path),
            "--output",
            str(output),
            "--source-root",
            str(source_root),
        ]
        log = (directory / "replay.log").open("ab")
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=self._script_env(),
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            self._process = process
            while True:
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                    break
                except TimeoutError:
                    done = sum(1 for _ in output.open(encoding="utf-8"))
                    if done != job["progress"]["completed"]:
                        job["progress"]["completed"] = done
                        self._store.save(job)
        finally:
            log.close()
            self._process = None
        self._check()
        if process.returncode != 0:
            raise RuntimeError(
                f"la repetición terminó con código {process.returncode}; ver replay.log"
            )
        results: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                results[row["case_id"]][row["setting"]].append(row)
        job["progress"]["completed"] = total
        self._store.save(job)
        return {key: dict(value) for key, value in results.items()}
