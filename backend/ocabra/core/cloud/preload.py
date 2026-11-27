"""Replicate our models onto a cloud node: download there, then load.

A node launched from the cloud panel can carry a list of *our* model ids. Once
it has joined, the cloud watchdog walks each one through a short list of
*steps* run with the node's own API over the tunnel:

* llama.cpp — ``download`` (same Ollama tag or Hugging Face GGUF as ours, with
  our settings minus what only makes sense on this machine; the node registers
  it itself) → ``load``.
* diffusers — ``backend`` (install diffusers on the node if it isn't) →
  ``download`` of exactly the files we have of each Hugging Face repo (the base
  pipeline without the transformer weights we don't use, the GGUF transformer)
  → ``register`` with the paths made relative to the node's models directory →
  ``load``.

An item's ``status`` is what the panel shows: ``pending``, ``installing``,
``downloading``, ``loading``, ``loaded`` or ``error``.

The node registers the model under the **same id** as ours, so the federation
routes requests for it there when we don't have it loaded.
"""

from __future__ import annotations

import asyncio
import copy
import os
import re
from pathlib import Path
from typing import Any

import httpx

from ocabra.core.cloud.manager import CloudError

# Settings tied to *our* GPU layout: meaningless (or harmful) on another box.
_HOST_SPECIFIC = ("disabled_gpus", "main_gpu", "tensor_split", "preferred_gpus", "gpu_indices")
# Left free on the node's GPU for the CUDA context and the projector/draft.
GPU_MARGIN_MB = 1500
MIN_CONTEXT = 32768
# First node version whose download API takes exact files into a folder.
EXACT_FILES_MIN_VERSION = (0, 87, 0)
# First node image that starts its own Ollama (backends/ollama_daemon.py).
OLLAMA_MIN_VERSION = (0, 144, 0)
_HF_CACHE_RE = re.compile(r"/models--([^/]+)/snapshots/([^/]+)(?:/(.*))?$")


_SPLIT_GGUF_RE = re.compile(r"-0*1-of-0*([2-9]|\d{2,})\.gguf$", re.IGNORECASE)


def _hf_file_from_path(ref: str, path: str) -> str | None:
    """The file inside the repo for a GGUF we downloaded from Hugging Face.

    Downloads land in ``huggingface/<org>--<repo>--<stem>/<file in repo>``;
    older ones only recorded that path (``source: local``), not ``file``.
    """
    repo, _, stem = ref.partition("::")
    folder = f"{repo.replace('/', '--')}--{stem}/"
    marker = f"/huggingface/{folder}"
    if marker not in path:
        return None
    return path.split(marker, 1)[1] or None


def download_source(artifact: dict[str, Any] | None) -> dict[str, Any]:
    """``{"source", "model_ref"[, "artifact"]}`` to fetch *artifact* elsewhere."""
    art = artifact or {}
    ref = str(art.get("ref") or "")
    if art.get("source") == "ollama" and ref:
        return {"source": "ollama", "model_ref": ref}
    file = art.get("file")
    if "::" in ref and not file and art.get("path"):
        file = _hf_file_from_path(ref, str(art["path"]))
    if "::" in ref and file:
        if _SPLIT_GGUF_RE.search(str(file)):
            raise CloudError(
                "Este modelo no se puede replicar todavía: es un GGUF partido en varios "
                "ficheros y la descarga del nodo solo baja uno",
                400,
            )
        repo = ref.split("::", 1)[0]
        return {"source": "huggingface", "model_ref": repo, "artifact": str(file)}
    raise CloudError(
        "Este modelo no se puede replicar: sus ficheros solo existen en este servidor "
        "(no vienen de Ollama ni de Hugging Face)",
        400,
    )


def fit_llama_cpp_config(cfg: dict[str, Any], gpu_total_mb: int | None) -> dict[str, Any]:
    """Our llama.cpp settings, adapted to a node with *gpu_total_mb* of VRAM.

    Drops host-specific keys. If our explicit VRAM reservation doesn't fit the
    node's GPU (ours fills a 3090 to the last MB; a 4090 reports slightly less),
    it is capped and the context halved so the KV cache shrinks with it.
    """
    out = {k: v for k, v in copy.deepcopy(cfg or {}).items() if k not in _HOST_SPECIFIC}
    estimate = out.get("vram_estimate_mb")
    if gpu_total_mb and isinstance(estimate, (int, float)):
        budget = int(gpu_total_mb) - GPU_MARGIN_MB
        if estimate > budget:
            out["vram_estimate_mb"] = max(1024, budget)
            ctx = out.get("ctx_size")
            if isinstance(ctx, int) and ctx > MIN_CONTEXT:
                out["ctx_size"] = max(MIN_CONTEXT, ctx // 2)
    return out


def replication_request(
    model_row: Any,
    gpu_total_mb: int | None,
    backend: str | None = None,
    gpus: list[int] | None = None,
) -> tuple[str, dict[str, Any]]:
    """``(node_model_id, download_body)`` to replicate *model_row* on a node.

    *backend* serves the same file with another engine than *model_row*'s
    (``ollama/gemma4:12b`` from our llama.cpp deployment): our settings are
    only carried over when they are that engine's.
    """
    from ocabra.core.model_ref import LLAMA_CPP_FAMILY, build_model_ref

    source_backend = str(model_row.backend_type)
    target = backend or source_backend
    if target not in LLAMA_CPP_FAMILY and target != "ollama":
        raise CloudError(
            f"Por ahora solo se replican modelos de llama.cpp y Ollama (este es {target})", 400
        )
    body = download_source(model_row.artifact)
    if target == "ollama" and body["source"] != "ollama":
        raise CloudError(
            "Este modelo no se puede servir con Ollama en el pod: su fichero no viene de "
            "un tag de Ollama",
            400,
        )
    if target == "ollama":
        # The node registers the tag as is; Ollama reads the manifest itself.
        body.pop("artifact", None)
    model_key = str(model_row.model_key or str(model_row.model_id).split("/", 1)[-1])
    extra = model_row.extra_config if isinstance(model_row.extra_config, dict) else {}
    if target == "ollama":
        own = extra.get("ollama") if source_backend == "ollama" else None
        section = (
            {k: v for k, v in own.items() if k not in _HOST_SPECIFIC}
            if isinstance(own, dict)
            else {}
        )
        extra_config = {"ollama": section} if section else {}
    else:
        own = extra.get("llama_cpp") if source_backend in LLAMA_CPP_FAMILY else None
        llama_cfg = fit_llama_cpp_config(own or {}, gpu_total_mb)
        extra_config = {"llama_cpp": llama_cfg} if llama_cfg else {}
    body["register_config"] = {
        "backend_type": target,
        "model_key": model_key,
        "extra_config": extra_config,
    }
    if gpus and target in LLAMA_CPP_FAMILY:
        # Como al descargar en el hub: el nodo planea sus perfiles (el mayor
        # contexto que cabe) con llama-fit sobre sus propias GPUs, en vez de
        # cargar con nuestro contexto, pensado para otras tarjetas.
        body["register_config"]["placement"] = {"gpus": list(gpus), "cpu": False}
    return build_model_ref(target, model_key), body


def _hf_cache_origin(path: str | Path) -> tuple[str, str, str] | None:
    """``(repo_id, revision, path_in_repo)`` if *path* lives in a HF cache snapshot."""
    # Resolve the folder, not the file: a snapshot's files are symlinks into
    # ``blobs/<sha>``, which says nothing about where they sit in the repo.
    path = str(path).rstrip("/")
    if os.path.isdir(path):
        resolved = os.path.realpath(path)
    else:
        resolved = os.path.join(os.path.realpath(os.path.dirname(path)), os.path.basename(path))
    m = _HF_CACHE_RE.search(resolved)
    if not m:
        return None
    repo = m.group(1).replace("--", "/", 1)
    return repo, m.group(2), m.group(3) or ""


def _tree_files(root: str) -> tuple[list[str], int]:
    """Relative file paths under *root* (following the cache's symlinks) and their size."""
    files: list[str] = []
    total = 0
    for dirpath, _dirs, names in os.walk(root, followlinks=True):
        for name in names:
            full = os.path.join(dirpath, name)
            files.append(os.path.relpath(full, root))
            try:
                total += os.stat(full).st_size
            except OSError:
                pass
    return sorted(files), total


def _absolute_paths(value: Any, prefix: str = "") -> list[str]:
    """Keys of *value* (nested) holding an absolute path."""
    out: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            out += _absolute_paths(v, f"{prefix}{k}.")
    elif isinstance(value, str) and value.startswith("/"):
        out.append(prefix.rstrip("."))
    return out


def _diffusers_plan(model_row: Any, models_dir: str) -> dict[str, Any]:
    """Steps to rebuild one of our diffusers models on a node (see module doc)."""
    art = model_row.artifact if isinstance(model_row.artifact, dict) else {}
    folder = str(art.get("ref") or str(model_row.model_id).split("/", 1)[-1])
    local_only = CloudError(
        "Este modelo no se puede replicar: sus ficheros no vienen de un repo de "
        "Hugging Face (están solo en este servidor)",
        400,
    )
    tree = os.path.join(models_dir, folder)
    origin = _hf_cache_origin(tree) if os.path.isdir(tree) else None
    if origin is None or origin[2]:
        raise local_only
    repo, revision, _ = origin
    files, size = _tree_files(tree)
    if not files:
        raise local_only

    extra = copy.deepcopy(
        model_row.extra_config if isinstance(model_row.extra_config, dict) else {}
    )
    for key in _HOST_SPECIFIC:
        extra.pop(key, None)
    section = extra.get("diffusers") if isinstance(extra.get("diffusers"), dict) else {}
    for key in _HOST_SPECIFIC:
        section.pop(key, None)
    steps: list[dict[str, Any]] = [{"kind": "backend", "backend": "diffusers"}]
    steps.append(
        {
            "kind": "download",
            "label": repo,
            "body": {
                "source": "huggingface",
                "model_ref": repo,
                "revision": revision,
                "allow_patterns": files,
                "local_dir": folder,
                "register": False,
            },
        }
    )
    gguf = section.get("gguf_transformer_path")
    if gguf:
        g_origin = _hf_cache_origin(gguf)
        if g_origin is None or not g_origin[2]:
            raise CloudError(
                "Este modelo no se puede replicar: su transformer GGUF está solo en este servidor",
                400,
            )
        g_repo, g_rev, g_file = g_origin
        g_dir = f"huggingface/{g_repo.replace('/', '--')}"
        try:
            size += os.stat(gguf).st_size
        except OSError:
            pass
        steps.append(
            {
                "kind": "download",
                "label": g_repo,
                "body": {
                    "source": "huggingface",
                    "model_ref": g_repo,
                    "revision": g_rev,
                    "allow_patterns": [g_file],
                    "local_dir": g_dir,
                    "register": False,
                },
            }
        )
        section["gguf_transformer_path"] = f"{g_dir}/{g_file}"
    if section:
        extra["diffusers"] = section
    leftovers = _absolute_paths(extra)
    if leftovers:
        raise CloudError(
            "Este modelo no se puede replicar: apunta a ficheros de este servidor "
            f"({', '.join(leftovers)})",
            400,
        )
    steps.append(
        {
            "kind": "register",
            "body": {
                "model_id": str(model_row.model_id),
                "backend_type": "diffusers",
                "display_name": model_row.display_name or None,
                "load_policy": "on_demand",
                "extra_config": extra or None,
            },
        }
    )
    steps.append({"kind": "load"})
    return {
        "node_model_id": str(model_row.model_id),
        "steps": steps,
        "size_bytes": size,
        "min_node_version": EXACT_FILES_MIN_VERSION,
    }


def plan_replication(
    model_row: Any,
    gpu_total_mb: int | None,
    models_dir: str | None = None,
    backend: str | None = None,
    gpus: list[int] | None = None,
) -> dict[str, Any]:
    """``{node_model_id, steps, size_bytes, min_node_version}`` for *model_row*.

    *backend*: serve it on the node with that engine instead of the row's.
    Touches the filesystem (diffusers trees): call it off the event loop.
    Raises :class:`CloudError` (400) with the reason when it can't be copied.
    """
    if str(model_row.backend_type) == "diffusers" and backend in (None, "diffusers"):
        from ocabra.config import settings

        return _diffusers_plan(model_row, models_dir or settings.models_dir)
    node_model_id, body = replication_request(model_row, gpu_total_mb, backend, gpus)
    return {
        "node_model_id": node_model_id,
        "steps": [{"kind": "download", "body": body}, {"kind": "load"}],
        "size_bytes": None,
        # Ollama on a pod needs a node that brings its own daemon.
        "min_node_version": OLLAMA_MIN_VERSION if node_model_id.startswith("ollama/") else None,
    }


REPLICABLE_BACKENDS = ("diffusers", "ollama")


async def load_model_rows(session_factory: Any, model_ids: list[str]) -> dict[str, Any]:
    import sqlalchemy as sa

    from ocabra.db.model_config import ModelConfig

    if not model_ids:
        return {}
    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    sa.select(ModelConfig).where(ModelConfig.model_id.in_(model_ids))
                )
            )
            .scalars()
            .all()
        )
    return {r.model_id: r for r in rows}


async def replication_rows(
    session_factory: Any, model_ids: list[str]
) -> dict[str, tuple[Any, str | None]]:
    """``{model_id: (row, backend)}``: what to copy each requested model from.

    An id we have is its own row. ``<backend>/<model>`` with no such
    deployment here (a benchmark asking for an engine we don't serve that
    model with) comes from the model's default deployment, served on the
    node with that *backend*.
    """
    import sqlalchemy as sa

    from ocabra.db.model_config import LogicalModel, ModelConfig

    rows = await load_model_rows(session_factory, model_ids)
    out: dict[str, tuple[Any, str | None]] = {mid: (row, None) for mid, row in rows.items()}
    missing = [mid for mid in model_ids if mid not in rows and "/" in mid]
    if not missing:
        return out
    async with session_factory() as session:
        for mid in missing:
            backend, _, key = mid.partition("/")
            logical = await session.get(LogicalModel, key)
            if logical is None:
                continue
            siblings = (
                (await session.execute(sa.select(ModelConfig).where(ModelConfig.model_key == key)))
                .scalars()
                .all()
            )
            source = next(
                (r for r in siblings if r.backend_type == logical.default_backend),
                siblings[0] if siblings else None,
            )
            if source is not None:
                out[mid] = (source, backend)
    return out


async def validate_models(session_factory: Any, model_ids: list[str]) -> None:
    """Fail fast (before renting anything) on unknown or non-replicable models."""
    for reason in (await replication_problems(session_factory, model_ids)).values():
        raise CloudError(reason, 400)


async def replication_problems(session_factory: Any, model_ids: list[str]) -> dict[str, str]:
    """``{model_id: why}`` for the models that can't be copied to a node."""
    rows = await replication_rows(session_factory, model_ids)
    problems: dict[str, str] = {}
    for mid in model_ids:
        found = rows.get(mid)
        if found is None:
            problems[mid] = f"Modelo desconocido: {mid}"
            continue
        row, backend = found
        try:
            await asyncio.to_thread(plan_replication, row, None, None, backend)
        except CloudError as exc:
            problems[mid] = str(exc)
    return problems


async def required_cuda(session_factory: Any, model_ids: list[str]) -> str | None:
    """Newest CUDA the prebuilt backends these models need are built for.

    The node installs backends prebuilt when published, with production's
    exact versions — our torch is a CUDA 13 build, which a host whose driver
    stops at 12.8 can't run. Best effort: ``None`` if nothing needs it or the
    registry can't be asked.
    """
    from ocabra.config import settings
    from ocabra.core.oci_registry import cuda_tuple, inspect_backend

    if not model_ids or not settings.backend_oci_repository:
        return None
    rows = await replication_rows(session_factory, model_ids)
    backends: set[str] = set()
    for row, backend in rows.values():
        try:
            plan = await asyncio.to_thread(plan_replication, row, None, None, backend)
        except CloudError:
            continue
        backends |= {s["backend"] for s in plan["steps"] if s["kind"] == "backend"}
    best: str | None = None
    for backend in sorted(backends):
        try:
            found = await inspect_backend(settings.backend_oci_repository, backend)
        except Exception:  # noqa: BLE001 — the node falls back to source anyway
            continue
        need = found.labels.get("org.ocabra.backend.cuda") if found else None
        if need and (best is None or cuda_tuple(need) > cuda_tuple(best)):
            best = need
    return best


def _node_gpu_total(peer: Any) -> int | None:
    gpus = getattr(peer, "gpus", None) or []
    totals = [int(g.get("total_vram_mb") or 0) for g in gpus if isinstance(g, dict)]
    return max(totals) if totals else None


def _node_gpu_indices(peer: Any) -> list[int]:
    gpus = getattr(peer, "gpus", None) or []
    out: list[int] = []
    for position, gpu in enumerate(gpus):
        if isinstance(gpu, dict):
            index = gpu.get("index")
            out.append(int(index) if isinstance(index, int) else position)
    return out


def _node_version(peer: Any) -> tuple[int, ...] | None:
    session = getattr(peer, "tunnel_session", None)
    raw = str(getattr(session, "remote_info", {}).get("version") or "") if session else ""
    nums = re.findall(r"\d+", raw)[:3]
    return tuple(int(n) for n in nums) if nums else None


ACTIVE = ("installing", "downloading", "registering", "loading")
# Lo que dice el nodo cuando no queda el mínimo libre para empezar una descarga
# (api/internal/downloads.py).
DISK_FULL_MARK = "insufficient disk space"
# Cortes de transporte seguidos (túnel, conexión) antes de dar un elemento por fallido.
MAX_TRANSIENT_FAILURES = 5


def _load_task(fm: Any, peer: Any, node_model_id: str) -> asyncio.Task:
    return asyncio.create_task(
        fm.proxy_request(peer, f"/ocabra/models/{node_model_id}/load", {}, {}, timeout=900),
        name=f"cloud-preload-load-{node_model_id}",
    )


async def _start_step(fm: Any, peer: Any, item: dict, tasks: dict, key: tuple) -> bool:
    """Begin the item's current step. ``True`` when it finished on the spot."""
    step = item["steps"][item["step"]]
    kind = step["kind"]
    item.pop("progress", None)
    item["phase"] = step.get("label")
    if kind == "backend":
        backend = step["backend"]
        resp = await fm.proxy_get(peer, f"/ocabra/backends/{backend}")
        if resp.status_code >= 400:
            raise CloudError(f"El nodo no conoce el backend {backend}: {resp.text[:200]}", 502)
        if resp.json().get("install_status") == "installed":
            return True
        item.update(status="installing", phase=f"instalando {backend}")
        # The install runs detached on the node: read the first event (so a
        # refusal shows up) and hang up; the next ticks poll its state.
        install = await fm.proxy_raw(
            peer,
            "POST",
            f"/ocabra/backends/{backend}/install",
            b'{"method": "auto"}',
            {"content-type": "application/json"},
        )
        try:
            if install.status_code >= 400 and install.status_code != 409:
                body = (await install.aread()).decode(errors="replace")
                raise CloudError(f"El nodo no pudo instalar {backend}: {body[:200]}", 502)
            async for _chunk in install.aiter_bytes():
                break
        finally:
            await install.aclose()
        return False
    if kind == "download":
        resp = await fm.proxy_request(peer, "/ocabra/downloads", step["body"], {})
        if resp.status_code >= 400:
            raise CloudError(f"El nodo rechazó la descarga: {resp.text[:200]}", 502)
        item.update(status="downloading", job_id=resp.json().get("job_id"), progress=0)
        return False
    if kind == "register":
        model_id = step["body"]["model_id"]
        existing = await fm.proxy_get(peer, f"/ocabra/models/{model_id}")
        if existing.status_code == 200:
            return True
        resp = await fm.proxy_request(peer, "/ocabra/models", step["body"], {})
        if resp.status_code >= 400:
            raise CloudError(f"El nodo no pudo registrar el modelo: {resp.text[:200]}", 502)
        return True
    if kind == "load":
        item["status"] = "loading"
        tasks[key] = _load_task(fm, peer, item["node_model_id"])
        return False
    raise CloudError(f"Paso desconocido: {kind}", 500)


async def _poll_step(fm: Any, peer: Any, item: dict, tasks: dict, key: tuple) -> bool:
    """Check the item's running step. ``True`` once it is done."""
    step = item["steps"][item["step"]]
    kind = step["kind"]
    if kind == "backend":
        resp = await fm.proxy_get(peer, f"/ocabra/backends/{step['backend']}")
        if resp.status_code >= 400:
            raise CloudError(f"No se pudo consultar la instalación: {resp.text[:200]}", 502)
        state = resp.json()
        status = state.get("install_status")
        if status == "installed":
            return True
        if status == "error":
            raise CloudError(
                f"Falló la instalación de {step['backend']}: {state.get('error')}", 502
            )
        if status == "not_installed":
            # Lost (the node restarted mid-install): ask again.
            return await _start_step(fm, peer, item, tasks, key)
        if state.get("install_progress") is not None:
            item["progress"] = round(float(state["install_progress"]), 1)
        return False
    if kind == "download":
        resp = await fm.proxy_get(peer, f"/ocabra/downloads/{item['job_id']}")
        if resp.status_code >= 400:
            raise CloudError(f"No se pudo consultar la descarga: {resp.text[:200]}", 502)
        job = resp.json()
        item["progress"] = round(float(job.get("progress_pct") or 0), 1)
        if job.get("status") in ("failed", "error", "cancelled"):
            raise CloudError(f"La descarga falló: {job.get('error') or job.get('status')}", 502)
        return job.get("status") == "completed"
    if kind == "load":
        task = tasks.get(key)
        if task is None:
            # Lost across a restart: ask again (loading an already loaded
            # model is a no-op on the node).
            tasks[key] = _load_task(fm, peer, item["node_model_id"])
            return False
        if not task.done():
            return False
        tasks.pop(key, None)
        resp = task.result()
        data = resp.json() if resp.content else {}
        if resp.status_code >= 400 or data.get("status") != "loaded":
            detail = data.get("detail") or data.get("error_message") or resp.text[:200]
            # Someone loaded it by hand meanwhile (the panel of the node):
            # that is what we wanted.
            if "already loaded" in str(detail).lower():
                return True
            raise CloudError(f"No se pudo cargar: {detail}", 502)
        return True
    return True


async def drive(
    fm: Any, peer: Any, items: list[dict[str, Any]], tasks: dict
) -> list[dict[str, Any]]:
    """Advance the preload one tick. Returns the updated list.

    Manual preloads are serial. Download-only benchmark items can occupy
    two slots; a finished download frees its slot in this same tick. Steps that
    finish immediately (backend already installed, model already registered)
    chain within the same tick; loads can take minutes, so they run as
    background tasks tracked in *tasks*.
    """
    items = [dict(i) for i in items]
    running = [i for i in items if i["status"] in ACTIVE]
    pending = [i for i in items if i["status"] == "pending"]
    occupied = 0
    for item in running + pending:
        parallel = item.get("benchmark_prefetch")
        if occupied and (not parallel or occupied >= 2):
            break
        await _drive_item(fm, peer, item, tasks)
        if not parallel:
            break
        if item["status"] in (*ACTIVE, "pending"):
            occupied += 1
    return items


async def _drive_item(fm: Any, peer: Any, active: dict, tasks: dict) -> None:
    key = (peer.peer_id, active["model_id"])
    before = copy.deepcopy(active)
    try:
        if active["status"] == "pending" or "steps" not in active:
            if active.get("recipe_steps"):
                # A recipe captured from a pod (core/cloud/recipes.py): its
                # own steps, no model of ours behind it.
                plan = {
                    "steps": copy.deepcopy(active["recipe_steps"]),
                    "node_model_id": active["model_id"],
                }
            else:
                rows = await replication_rows(fm.session_factory, [active["model_id"]])
                found = rows.get(active["model_id"])
                if found is None:
                    raise CloudError("El modelo ya no existe aquí", 400)
                row, backend = found
                plan = await asyncio.to_thread(
                    plan_replication,
                    row,
                    _node_gpu_total(peer),
                    None,
                    backend,
                    gpus=_node_gpu_indices(peer),
                )
                need = plan.get("min_node_version")
                have = _node_version(peer)
                if need and have is not None and have < tuple(need):
                    raise CloudError(
                        "El nodo es de una versión anterior que no sabe copiar este modelo "
                        f"(tiene {'.'.join(map(str, have))}); lánzalo de nuevo con la imagen actual",
                        409,
                    )
            steps = plan["steps"]
            if active.get("load") is False:
                # «Solo descargar»: on the node, ready, loaded when asked for.
                steps = [s for s in steps if s["kind"] != "load"]
            active.update(steps=steps, step=0, node_model_id=plan["node_model_id"])
            done = await _start_step(fm, peer, active, tasks, key)
        else:
            done = await _poll_step(fm, peer, active, tasks, key)
        while done:
            active["step"] += 1
            if active["step"] >= len(active["steps"]):
                if active.get("benchmark_prefetch"):
                    model = active["node_model_id"]
                    response = await fm.proxy_get(peer, f"/ocabra/models/{model}")
                    if response.status_code != 200:
                        raise CloudError(f"El modelo descargado no está registrado: {model}", 502)
                active.update(
                    status="loaded" if active.get("load") is not False else "ready", phase=None
                )
                active.pop("progress", None)
                break
            done = await _start_step(fm, peer, active, tasks, key)
    except (httpx.TransportError, httpx.TimeoutException) as exc:
        # El túnel se reconecta (p. ej. al rotar la conexión) y la petición en
        # vuelo cae con «federation tunnel closed». No es un fallo del modelo:
        # se deja el elemento como estaba y se reintenta en la próxima pasada.
        failures = int(before.get("transient_failures") or 0) + 1
        if failures >= MAX_TRANSIENT_FAILURES:
            tasks.pop(key, None)
            active.update(
                status="error",
                error=f"el nodo no respondió {failures} veces seguidas: {exc}"[:300],
            )
        else:
            active.clear()
            active.update(before, transient_failures=failures, last_transient_error=str(exc)[:200])
        return
    except Exception as exc:  # noqa: BLE001 — one model failing must not stop the rest
        tasks.pop(key, None)
        if active.get("benchmark_prefetch") and DISK_FULL_MARK in str(exc):
            # El nodo rechaza la descarga antes de empezarla (no deja nada a
            # medias): en una batería se borran los modelos ya medidos, así
            # que se vuelve a intentar cuando haya sitio.
            active.clear()
            active.update(
                {k: v for k, v in before.items() if k not in ("steps", "step", "job_id")},
                status="pending",
                phase="esperando espacio en disco",
                waiting_disk=True,
            )
            return
        active.update(status="error", error=str(exc)[:300])
        return
    active.pop("waiting_disk", None)
    active.pop("transient_failures", None)
    active.pop("last_transient_error", None)


async def replicable_models(
    session_factory: Any, model_manager: Any = None
) -> list[dict[str, Any]]:
    """Our models that a cloud node could carry, with why not when it can't.

    Loaded ones first (what is in use here is what you'd want more of), then
    by name. Uses the same rule as the preload itself, so the UI never offers
    something the launch would reject.
    """
    import sqlalchemy as sa

    from ocabra.core.model_ref import LLAMA_CPP_FAMILY
    from ocabra.db.model_config import ModelConfig

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    sa.select(ModelConfig).where(
                        ModelConfig.backend_type.in_((*LLAMA_CPP_FAMILY, *REPLICABLE_BACKENDS))
                    )
                )
            )
            .scalars()
            .all()
        )
    states: dict[str, Any] = {}
    if model_manager is not None:
        try:
            for st in await model_manager.list_states():
                states[st.model_id] = st
        except Exception:  # noqa: BLE001 — sizes/status are a nicety
            states = {}

    def _plans() -> list[tuple[dict | None, str | None]]:
        out_: list[tuple[dict | None, str | None]] = []
        for row in rows:
            try:
                out_.append((plan_replication(row, None), None))
            except CloudError as exc:
                out_.append((None, str(exc)))
        return out_

    plans = await asyncio.to_thread(_plans)
    # ModelState carries no size: the Models API works it out per model.
    sizes: dict[str, int | None] = {}
    try:
        from ocabra.api.internal.models import _resolve_disk_size_bytes

        for mid, st in states.items():
            if str(getattr(st, "backend_type", "")) in LLAMA_CPP_FAMILY:
                sizes[mid] = await _resolve_disk_size_bytes(st, {})
    except Exception:  # noqa: BLE001 — sizes are a nicety
        pass
    out = []
    for row, (plan, reason) in zip(rows, plans, strict=True):
        st = states.get(row.model_id)
        replicable = plan is not None
        status = getattr(getattr(st, "status", None), "value", None)
        out.append(
            {
                "model_id": row.model_id,
                "display_name": getattr(st, "display_name", None)
                or row.display_name
                or row.model_id,
                "backend_type": str(row.backend_type),
                # What the node will download (diffusers: only the files we
                # use, not the whole repo), else what it takes on our disk.
                "size_bytes": (plan or {}).get("size_bytes") or sizes.get(row.model_id),
                "loaded": status == "loaded",
                "replicable": replicable,
                "reason": reason,
            }
        )
    out.sort(key=lambda m: (not m["replicable"], not m["loaded"], m["display_name"].lower()))
    return out
