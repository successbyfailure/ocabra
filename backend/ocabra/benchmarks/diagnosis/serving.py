"""Lo que de verdad ve el backend: prompt renderizado, parámetros y contexto.

Todo se consulta al llama-server vivo (``/props``, ``/apply-template``,
``/tokenize``) y a las mismas funciones de la pipeline que construyen el
cuerpo final (``merge_profile_defaults`` → ``apply_declared_output_limit`` →
``to_backend_body``). Lo que el cuerpo no fija lo pone el llama-server con sus
propios valores por defecto, así que los parámetros efectivos son esos
valores con el cuerpo encima.

La plantilla oficial se renderiza con jinja2 en un subproceso aislado (``uv
run --with jinja2``): el api no lleva jinja2 ni transformers.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ocabra.benchmarks.models import bench_profile
from ocabra.config import settings

# Lo que se compara entre lo efectivo y lo recomendado.
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "dry_multiplier",
)
OTHER_KEYS = (
    "max_tokens",
    "reasoning_effort",
    "reasoning_budget_tokens",
    "chat_template_kwargs",
    "reasoning_format",
)
_LLAMA_SERVERS = ("llama_cpp", "llamacpp_mks")
SCRIPTS = Path(__file__).resolve().parent / "scripts"


@dataclass
class LiveWorker:
    worker_key: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


async def find_loaded_worker(model_manager: Any, base_model_id: str) -> LiveWorker | None:
    """El worker cargado del modelo (base o uno de sus presets ``::``)."""
    from ocabra.core.model_manager import ModelStatus

    for state in await model_manager.list_states(include_derived=True):
        key = str(state.model_id)
        if key != base_model_id and not key.startswith(f"{base_model_id}::"):
            continue
        info = getattr(state, "worker_info", None)
        if state.status == ModelStatus.LOADED and info is not None and info.port:
            return LiveWorker(worker_key=key, port=int(info.port))
    return None


class WorkerProbe:
    """Consultas de solo lectura al llama-server de un worker."""

    def __init__(self, worker: LiveWorker, client: httpx.AsyncClient) -> None:
        self.worker = worker
        self._client = client
        self._props: dict[str, Any] | None = None

    async def props(self) -> dict[str, Any]:
        if self._props is None:
            response = await self._client.get(f"{self.worker.base_url}/props", timeout=10)
            response.raise_for_status()
            self._props = response.json()
        return self._props

    async def apply_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {"messages": messages}
        if tools:
            body["tools"] = tools
        if chat_template_kwargs:
            body["chat_template_kwargs"] = chat_template_kwargs
        response = await self._client.post(
            f"{self.worker.base_url}/apply-template", json=body, timeout=30
        )
        response.raise_for_status()
        return str(response.json().get("prompt") or "")

    async def tokenize(self, text: str, *, add_special: bool) -> list[int]:
        response = await self._client.post(
            f"{self.worker.base_url}/tokenize",
            json={"content": text, "add_special": add_special, "parse_special": True},
            timeout=30,
        )
        response.raise_for_status()
        tokens = response.json().get("tokens") or []
        return [int(item if not isinstance(item, dict) else item.get("id", 0)) for item in tokens]

    async def server_defaults(self) -> dict[str, Any]:
        params = ((await self.props()).get("default_generation_settings") or {}).get("params") or {}
        defaults = {key: params.get(key) for key in SAMPLING_KEYS if key in params}
        if "n_predict" in params:
            defaults["max_tokens"] = params["n_predict"]
        return defaults

    async def context(self) -> dict[str, int]:
        props = await self.props()
        settings_ = props.get("default_generation_settings") or {}
        return {
            "n_ctx_slot": int(settings_.get("n_ctx") or 0),
            "slots": int(props.get("total_slots") or 0),
        }


# ── Parámetros efectivos ──────────────────────────────────────────────────


async def backend_body(
    app_state: Any,
    profile_id: str,
    client_body: dict[str, Any],
    messages: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], Any, Any]:
    """El cuerpo que la pipeline mandaría al backend para esta petición.

    Devuelve ``(cuerpo, perfil, estado)``. No envía nada.
    """
    from ocabra.core.inference.reasoning import to_backend_body
    from ocabra.core.inference.resolution import merge_profile_defaults
    from ocabra.core.output_limits import apply_declared_output_limit

    profile = await bench_profile(app_state.profile_registry, profile_id)
    if profile is None:
        raise LookupError(f"perfil '{profile_id}' no encontrado")
    state = await app_state.model_manager.get_state(profile.base_model_id)
    if state is None:
        raise LookupError(f"modelo '{profile.base_model_id}' no encontrado")
    body: dict[str, Any] = {"model": profile_id, "messages": messages or [], **client_body}
    if tools:
        body["tools"] = tools
    merged = merge_profile_defaults(profile, body)
    capabilities = getattr(state, "capabilities", None)
    context = int(getattr(capabilities, "context_length", 0) or 0)
    apply_declared_output_limit(
        profile,
        merged,
        context_length=context,
        fallback_limit=settings.default_max_output_tokens,
    )
    return to_backend_body(state, merged), profile, state


def effective_params(server_defaults: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    effective = dict(server_defaults)
    for key in (*SAMPLING_KEYS, *OTHER_KEYS):
        if key in body:
            effective[key] = body[key]
    if "max_completion_tokens" in body and "max_tokens" not in body:
        effective["max_tokens"] = body["max_completion_tokens"]
    return effective


def sampling_differences(
    effective: dict[str, Any], recommended: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Diferencias de muestreo que pueden cambiar el resultado."""
    diffs: dict[str, dict[str, Any]] = {}
    for key, wanted in recommended.items():
        got = effective.get(key)
        if got is None:
            continue
        try:
            same = abs(float(got) - float(wanted)) < 1e-3
        except (TypeError, ValueError):
            same = got == wanted
        if not same:
            diffs[key] = {"efectivo": got, "recomendado": wanted}
    return diffs


# ── Plantilla oficial ─────────────────────────────────────────────────────


def benchmark_root() -> Path:
    """Raíz de los benchmarks (cachés de uv compartidas con los runners)."""
    return Path(os.path.abspath(settings.benchmark_dir))


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("hace falta uv para renderizar la plantilla oficial")
    return uv


async def render_official(
    template: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    bos_token: str | None,
    eos_token: str | None,
    kwargs: dict[str, Any] | None = None,
) -> str:
    """Renderiza la plantilla oficial de HF como lo haría ``apply_chat_template``."""
    root = benchmark_root()
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(root / "uv" / "cache")
    env["UV_PYTHON_INSTALL_DIR"] = str(root / "uv" / "python")
    env.pop("VIRTUAL_ENV", None)
    payload = json.dumps(
        {
            "template": template,
            "messages": messages,
            "tools": tools,
            "bos_token": bos_token or "",
            "eos_token": eos_token or "",
            "kwargs": kwargs or {},
        }
    ).encode()
    process = await asyncio.create_subprocess_exec(
        _uv(), "run", "--no-project", "--with", "jinja2>=3.1", "python", "-I",
        str(SCRIPTS / "render_template.py"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )  # fmt: skip
    stdout, stderr = await asyncio.wait_for(process.communicate(payload), timeout=180)
    if process.returncode != 0:
        raise RuntimeError(stderr.decode(errors="replace").strip()[-400:] or "render falló")
    return json.loads(stdout.decode())["prompt"]


def _strip_bos(text: str, bos: str | None) -> str:
    if bos:
        while text.startswith(bos):
            text = text[len(bos) :]
    return text


def compare_prompts(
    live: str, official: str, *, live_bos: str | None, official_bos: str | None
) -> dict[str, Any]:
    """Compara los dos prompts. El BOS se mira aparte (lo pone el tokenizador)."""
    a = _strip_bos(live, live_bos)
    b = _strip_bos(official, official_bos)
    if a == b:
        return {"status": "identical", "diff": []}
    status = "whitespace" if "".join(a.split()) == "".join(b.split()) else "different"
    return {"status": status, "diff": _diff(a, b), "first_difference": first_difference(a, b)}


def first_difference(live: str, official: str, context: int = 60) -> dict[str, Any]:
    """Dónde empiezan a diferir, con contexto (las líneas del diff se cortan)."""
    index = next(
        (i for i, (x, y) in enumerate(zip(live, official, strict=False)) if x != y),
        min(len(live), len(official)),
    )
    start = max(0, index - context)
    return {
        "offset": index,
        "oficial": official[start : index + context],
        "servido": live[start : index + context],
    }


def _diff(a: str, b: str, limit: int = 40) -> list[str]:
    lines = list(
        difflib.unified_diff(
            b.splitlines(),
            a.splitlines(),
            "oficial (HF)",
            "servido (llama-server)",
            lineterm="",
            n=1,
        )
    )
    return [line[:300] for line in lines[:limit]]


def is_llama_server(backend_type: str | None) -> bool:
    return str(backend_type or "") in _LLAMA_SERVERS
