"""Replicate our models onto a cloud node: download there, then load.

A node launched from the cloud panel can carry a list of *our* model ids. Once
it has joined, the cloud watchdog drives each one through
``pending → downloading → loading → loaded`` (or ``error``) using the node's own
API over the tunnel: ``POST /ocabra/downloads`` with the same source as ours
(an Ollama tag or a Hugging Face GGUF) and the same llama.cpp settings, minus
what only makes sense on our machine, then ``POST /ocabra/models/{id}/load``.

The node registers the model under the **same id** as ours, so the federation
routes requests for it there when we don't have it loaded.
"""

from __future__ import annotations

import copy
from typing import Any

from ocabra.core.cloud.manager import CloudError

# Settings tied to *our* GPU layout: meaningless (or harmful) on another box.
_HOST_SPECIFIC = ("disabled_gpus", "main_gpu", "tensor_split", "preferred_gpus", "gpu_indices")
# Left free on the node's GPU for the CUDA context and the projector/draft.
GPU_MARGIN_MB = 1500
MIN_CONTEXT = 32768


def download_source(artifact: dict[str, Any] | None) -> dict[str, Any]:
    """``{"source", "model_ref"[, "artifact"]}`` to fetch *artifact* elsewhere."""
    art = artifact or {}
    ref = str(art.get("ref") or "")
    if art.get("source") == "ollama" and ref:
        return {"source": "ollama", "model_ref": ref}
    if "::" in ref and art.get("file"):
        repo = ref.split("::", 1)[0]
        return {"source": "huggingface", "model_ref": repo, "artifact": str(art["file"])}
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


def replication_request(model_row: Any, gpu_total_mb: int | None) -> tuple[str, dict[str, Any]]:
    """``(node_model_id, download_body)`` to replicate *model_row* on a node."""
    from ocabra.core.model_ref import LLAMA_CPP_FAMILY, build_model_ref

    backend = str(model_row.backend_type)
    if backend not in LLAMA_CPP_FAMILY:
        raise CloudError(
            f"Por ahora solo se replican modelos de llama.cpp (este es {backend})", 400
        )
    body = download_source(model_row.artifact)
    model_key = str(model_row.model_key or str(model_row.model_id).split("/", 1)[-1])
    extra = model_row.extra_config if isinstance(model_row.extra_config, dict) else {}
    llama_cfg = fit_llama_cpp_config(extra.get("llama_cpp") or {}, gpu_total_mb)
    body["register_config"] = {
        "backend_type": backend,
        "model_key": model_key,
        "extra_config": {"llama_cpp": llama_cfg} if llama_cfg else {},
    }
    return build_model_ref(backend, model_key), body


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


async def validate_models(session_factory: Any, model_ids: list[str]) -> None:
    """Fail fast (before renting anything) on unknown or non-replicable models."""
    rows = await load_model_rows(session_factory, model_ids)
    for mid in model_ids:
        row = rows.get(mid)
        if row is None:
            raise CloudError(f"Modelo desconocido: {mid}", 400)
        replication_request(row, None)


def _node_gpu_total(peer: Any) -> int | None:
    gpus = getattr(peer, "gpus", None) or []
    totals = [int(g.get("total_vram_mb") or 0) for g in gpus if isinstance(g, dict)]
    return max(totals) if totals else None


async def drive(
    fm: Any, peer: Any, items: list[dict[str, Any]], tasks: dict
) -> list[dict[str, Any]]:
    """Advance each preload item one step. Returns the updated list.

    One model is worked on at a time (download, then load) so a node isn't
    asked to fill its disk and VRAM with everything at once. Loads can take
    minutes, so they run as background tasks tracked in *tasks*.
    """
    items = [dict(i) for i in items]
    active = next((i for i in items if i["status"] in ("downloading", "loading")), None)
    if active is None:
        active = next((i for i in items if i["status"] == "pending"), None)
    if active is None:
        return items

    key = (peer.peer_id, active["model_id"])
    try:
        if active["status"] == "pending":
            rows = await load_model_rows(fm.session_factory, [active["model_id"]])
            row = rows.get(active["model_id"])
            if row is None:
                raise CloudError("El modelo ya no existe aquí", 400)
            node_model_id, body = replication_request(row, _node_gpu_total(peer))
            resp = await fm.proxy_request(peer, "/ocabra/downloads", body, {})
            if resp.status_code >= 400:
                raise CloudError(f"El nodo rechazó la descarga: {resp.text[:200]}", 502)
            active.update(
                status="downloading",
                job_id=resp.json().get("job_id"),
                node_model_id=node_model_id,
                progress=0,
            )
        elif active["status"] == "downloading":
            resp = await fm.proxy_get(peer, f"/ocabra/downloads/{active['job_id']}")
            if resp.status_code >= 400:
                raise CloudError(f"No se pudo consultar la descarga: {resp.text[:200]}", 502)
            job = resp.json()
            active["progress"] = round(float(job.get("progress_pct") or 0), 1)
            if job.get("status") in ("failed", "error", "cancelled"):
                raise CloudError(f"La descarga falló: {job.get('error') or job.get('status')}", 502)
            if job.get("status") == "completed":
                import asyncio

                active["status"] = "loading"
                tasks[key] = asyncio.create_task(
                    fm.proxy_request(
                        peer, f"/ocabra/models/{active['node_model_id']}/load", {}, {}, timeout=900
                    ),
                    name=f"cloud-preload-load-{active['node_model_id']}",
                )
        elif active["status"] == "loading":
            task = tasks.get(key)
            if task is None:
                # Lost across a restart: ask again (loading an already loaded
                # model is a no-op on the node).
                import asyncio

                tasks[key] = asyncio.create_task(
                    fm.proxy_request(
                        peer, f"/ocabra/models/{active['node_model_id']}/load", {}, {}, timeout=900
                    )
                )
            elif task.done():
                tasks.pop(key, None)
                resp = task.result()
                data = resp.json() if resp.content else {}
                if resp.status_code >= 400 or data.get("status") != "loaded":
                    detail = data.get("detail") or data.get("error_message") or resp.text[:200]
                    raise CloudError(f"No se pudo cargar: {detail}", 502)
                active["status"] = "loaded"
    except Exception as exc:  # noqa: BLE001 — one model failing must not stop the rest
        tasks.pop(key, None)
        active.update(status="error", error=str(exc)[:300])
    return items


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
                    sa.select(ModelConfig).where(ModelConfig.backend_type.in_(LLAMA_CPP_FAMILY))
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
    out = []
    for row in rows:
        st = states.get(row.model_id)
        try:
            replication_request(row, None)
            replicable, reason = True, None
        except CloudError as exc:
            replicable, reason = False, str(exc)
        status = getattr(getattr(st, "status", None), "value", None)
        out.append(
            {
                "model_id": row.model_id,
                "display_name": getattr(st, "display_name", None)
                or row.display_name
                or row.model_id,
                "size_bytes": getattr(st, "disk_size_bytes", None),
                "loaded": status == "loaded",
                "replicable": replicable,
                "reason": reason,
            }
        )
    out.sort(key=lambda m: (not m["replicable"], not m["loaded"], m["display_name"].lower()))
    return out
