"""Model recipes for cloud nodes: what a pod had, so the next one can have it too.

A model downloaded and tuned by hand in a pod's panel (the 120B nemotron on
the RTX PRO 6000, 2026-11-17) only ever existed on that pod: when it was
terminated or its image updated, nobody knew where it came from or how it was
configured. The hub now asks each live node, every few minutes and right
before terminating or restarting it, for the models it downloaded and
registered, and keeps a recipe per model: the download (source, repo, file)
with the node's current config folded into its registration, and a load.
The launch dialog offers them under «Recientes en pods».

Models that also exist here are left out: the launch dialog already offers
those («De este nodo») and copies them from our own config.
"""

from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import structlog

logger = structlog.get_logger(__name__)

# Keys of a node's model config that only make sense on that machine.
_DROP_TOP = ("_runtime", "registry_metadata")
_HOST_KEYS = (
    "model_path",
    "mmproj_path",
    "disabled_gpus",
    "main_gpu",
    "tensor_split",
    "preferred_gpus",
    "gpu_indices",
)


def clean_extra(extra: dict[str, Any] | None) -> dict[str, Any]:
    """A node's ``extra_config`` without what is specific to that machine."""
    out = copy.deepcopy(extra or {})
    for key in _DROP_TOP:
        out.pop(key, None)
    for section in list(out.values()):
        if isinstance(section, dict):
            for key in _HOST_KEYS:
                section.pop(key, None)
            for key, value in list(section.items()):
                if isinstance(value, str) and value.startswith("/"):
                    section.pop(key, None)
    return out


def _source_label(job: dict[str, Any]) -> str:
    ref = str(job.get("model_ref") or "")
    artifact = job.get("artifact")
    return f"{job.get('source') or '?'}: {ref}" + (f" · {artifact}" if artifact else "")


def recipe_from_node(job: dict[str, Any], model: dict[str, Any]) -> dict[str, Any] | None:
    """The recipe for a model a node downloaded (*job*) and has now (*model*)."""
    rc = job.get("register_config")
    if not isinstance(rc, dict) or not job.get("model_ref"):
        return None
    register = {k: v for k, v in rc.items() if k != "placement"}
    register["extra_config"] = clean_extra(model.get("extra_config"))
    if model.get("load_policy"):
        register["load_policy"] = model["load_policy"]
    if model.get("display_name"):
        register["display_name"] = model["display_name"]
    body = {
        key: job[key]
        for key in ("source", "model_ref", "artifact", "revision", "allow_patterns", "local_dir")
        if job.get(key) is not None
    }
    body.update(register=True, register_config=register)
    return {
        "node_model_id": str(model["model_id"]),
        "display_name": model.get("display_name") or rc.get("display_name"),
        "backend_type": str(model.get("backend_type") or rc.get("backend_type") or "?"),
        "source": _source_label(job)[:512],
        "size_bytes": model.get("disk_size_bytes"),
        "steps": [
            {"kind": "download", "label": str(job["model_ref"]), "body": body},
            {"kind": "load"},
        ],
        "last_request_at": model.get("last_request_at"),
    }


async def _node_recipes(fm: Any, peer: Any) -> list[dict[str, Any]]:
    resp = await fm.proxy_get(peer, "/ocabra/downloads")
    if resp.status_code >= 400:
        return []
    jobs = resp.json()
    jobs = jobs.get("jobs", jobs) if isinstance(jobs, dict) else jobs
    out: dict[str, dict[str, Any]] = {}
    for job in jobs or []:
        if not isinstance(job, dict) or job.get("status") != "completed" or not job.get("register"):
            continue
        rc = job.get("register_config") or {}
        backend, key = rc.get("backend_type"), rc.get("model_key")
        if not backend or not key:
            continue
        model_id = f"{backend}/{key}"
        got = await fm.proxy_get(peer, f"/ocabra/models/{quote(model_id, safe='')}")
        if got.status_code != 200:
            continue  # deleted on the node since
        recipe = recipe_from_node(job, got.json())
        if recipe is not None:
            out[recipe["node_model_id"]] = recipe  # the latest job of a model wins
    return list(out.values())


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


async def snapshot(fm: Any, peer: Any, *, node_name: str, gpu_type_id: str | None) -> int:
    """Store the recipes of the models *peer* has. Returns how many."""
    import sqlalchemy as sa

    from ocabra.db.cloud import CloudModelRecipe
    from ocabra.db.model_config import ModelConfig

    recipes = await _node_recipes(fm, peer)
    if not recipes:
        return 0
    now = datetime.now(UTC)
    async with fm.session_factory() as session:
        ours = set(
            (
                await session.execute(
                    sa.select(ModelConfig.model_id).where(
                        ModelConfig.model_id.in_([r["node_model_id"] for r in recipes])
                    )
                )
            )
            .scalars()
            .all()
        )
        stored = 0
        for r in recipes:
            if r["node_model_id"] in ours:
                continue  # offered as one of our own models
            row = (
                await session.execute(
                    sa.select(CloudModelRecipe).where(
                        CloudModelRecipe.node_model_id == r["node_model_id"]
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                row = CloudModelRecipe(id=uuid.uuid4(), node_model_id=r["node_model_id"])
                session.add(row)
            row.display_name = r["display_name"]
            row.backend_type = r["backend_type"]
            row.source = r["source"]
            row.size_bytes = r["size_bytes"]
            row.steps = r["steps"]
            row.last_node_name = node_name
            row.last_gpu_type_id = gpu_type_id
            row.captured_at = now
            used = _parse_dt(r.get("last_request_at"))
            if used and (row.last_used_at is None or used > row.last_used_at):
                row.last_used_at = used
            stored += 1
        await session.commit()
    if stored:
        logger.info("cloud_recipes_captured", node=node_name, count=stored)
    return stored


def recipe_out(row: Any) -> dict[str, Any]:
    load_cfg: dict[str, Any] = {}
    for step in row.steps or []:
        if step.get("kind") == "download":
            extra = ((step.get("body") or {}).get("register_config") or {}).get(
                "extra_config"
            ) or {}
            load_cfg = extra.get("llama_cpp") or {}
    return {
        "id": str(row.id),
        "node_model_id": row.node_model_id,
        "display_name": row.display_name or row.node_model_id.split("/", 1)[-1],
        "backend_type": row.backend_type,
        "source": row.source,
        "size_bytes": row.size_bytes,
        "context": load_cfg.get("ctx_size"),
        "parallel": load_cfg.get("parallel_slots"),
        "vram_estimate_mb": load_cfg.get("vram_estimate_mb"),
        "load_policy": next(
            (
                ((s.get("body") or {}).get("register_config") or {}).get("load_policy")
                for s in row.steps or []
                if s.get("kind") == "download"
            ),
            None,
        ),
        "last_node_name": row.last_node_name,
        "last_gpu_type_id": row.last_gpu_type_id,
        "last_used_at": row.last_used_at,
        "captured_at": row.captured_at,
    }


async def list_recipes(session_factory: Any) -> list[dict[str, Any]]:
    """Most recently used (or captured) first."""
    import sqlalchemy as sa

    from ocabra.db.cloud import CloudModelRecipe

    async with session_factory() as session:
        rows = (await session.execute(sa.select(CloudModelRecipe))).scalars().all()
    rows = sorted(rows, key=lambda r: r.last_used_at or r.captured_at, reverse=True)
    return [recipe_out(r) for r in rows]


async def load_recipes(session_factory: Any, ids: list[str]) -> dict[str, Any]:
    import sqlalchemy as sa

    from ocabra.db.cloud import CloudModelRecipe

    parsed = []
    for rid in ids:
        try:
            parsed.append(uuid.UUID(str(rid)))
        except ValueError:
            continue
    if not parsed:
        return {}
    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    sa.select(CloudModelRecipe).where(CloudModelRecipe.id.in_(parsed))
                )
            )
            .scalars()
            .all()
        )
    return {str(r.id): r for r in rows}


async def delete_recipe(session_factory: Any, recipe_id: str) -> bool:
    from ocabra.db.cloud import CloudModelRecipe

    async with session_factory() as session:
        row = await session.get(CloudModelRecipe, uuid.UUID(recipe_id))
        if row is None:
            return False
        await session.delete(row)
        await session.commit()
    return True


def backend_item(backend: str) -> dict[str, Any]:
    """A preload entry that only installs *backend* on the node."""
    return {
        "model_id": f"backend:{backend}",
        "kind": "backend",
        "status": "pending",
        "recipe_steps": [{"kind": "backend", "backend": backend}],
        "load": False,
    }


def preload_item(row: Any, *, load: bool) -> dict[str, Any]:
    """A preload entry that runs the recipe's own steps (no model of ours behind)."""
    return {
        "model_id": row.node_model_id,
        "status": "pending",
        "recipe_id": str(row.id),
        "recipe_steps": row.steps,
        "load": load,
    }


# ── recipes made by hand: search in the launch dialog, edit before launching ──

LLAMA_BACKEND = "llamacpp_mks"
LLAMA_FAMILY = ("llamacpp_mks", "llama_cpp", "bitnet")
# Already in the ocabra-node image (backend/Dockerfile.node): nothing to install.
IMAGE_BACKENDS = ("llamacpp_mks",)


def _model_key(source: str, model_ref: str, artifact: str | None) -> str:
    """The id the node registers it under: the Ollama tag, or the GGUF file's
    name (one repo usually has several quantizations)."""
    if source == "ollama":
        return model_ref.strip().lower()
    stem = (artifact or model_ref).rsplit("/", 1)[-1]
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    return stem.strip().lower()


def _apply_config(
    register: dict[str, Any],
    *,
    context: int | None = None,
    parallel: int | None = None,
    load_policy: str | None = None,
    display_name: str | None = None,
) -> None:
    extra = register.setdefault("extra_config", {})
    llama = extra.setdefault("llama_cpp", {})
    if context is not None:
        if context > 0:
            llama["ctx_size"] = int(context)
            # A reservation computed for the old context would be wrong now:
            # the node sizes it again.
            llama.pop("vram_estimate_mb", None)
        else:
            llama.pop("ctx_size", None)
    if parallel is not None:
        if parallel > 0:
            llama["parallel_slots"] = int(parallel)
            llama.pop("vram_estimate_mb", None)
        else:
            llama.pop("parallel_slots", None)
    if not llama:
        extra.pop("llama_cpp", None)
    if load_policy:
        register["load_policy"] = load_policy
    if display_name:
        register["display_name"] = display_name


def new_recipe(
    *,
    source: str,
    model_ref: str,
    artifact: str | None,
    display_name: str | None,
    size_bytes: int | None,
    context: int | None,
    parallel: int | None,
    load_policy: str,
    backend_type: str = LLAMA_BACKEND,
) -> dict[str, Any]:
    """A recipe for a model picked in the launch dialog's search.

    Any backend the node can install: the download registers it there for
    *backend_type*, as Explore does here. Context and slots only mean
    something to llama.cpp."""
    key = _model_key(source, model_ref, artifact)
    if source == "huggingface" and not artifact:
        key = model_ref.rsplit("/", 1)[-1].strip().lower()
    llama = backend_type in LLAMA_FAMILY
    register: dict[str, Any] = {"backend_type": backend_type, "model_key": key}
    _apply_config(
        register,
        context=context if llama else None,
        parallel=parallel if llama else None,
        load_policy=load_policy,
        display_name=display_name or (model_ref if source == "ollama" else key),
    )
    body: dict[str, Any] = {"source": source, "model_ref": model_ref}
    if artifact:
        body["artifact"] = artifact
    body.update(register=True, register_config=register)
    return {
        "node_model_id": f"{backend_type}/{key}",
        "display_name": register["display_name"],
        "backend_type": backend_type,
        "source": _source_label({"source": source, "model_ref": model_ref, "artifact": artifact})[
            :512
        ],
        "size_bytes": size_bytes,
        "steps": [
            {"kind": "download", "label": model_ref, "body": body},
            {"kind": "load"},
        ],
    }


async def create_recipe(session_factory: Any, **fields: Any) -> dict[str, Any]:
    """Store a hand-made recipe (or update the one for the same model)."""
    import sqlalchemy as sa

    from ocabra.db.cloud import CloudModelRecipe

    r = new_recipe(**fields)
    async with session_factory() as session:
        row = (
            await session.execute(
                sa.select(CloudModelRecipe).where(
                    CloudModelRecipe.node_model_id == r["node_model_id"]
                )
            )
        ).scalar_one_or_none()
        if row is None:
            row = CloudModelRecipe(id=uuid.uuid4(), node_model_id=r["node_model_id"])
            session.add(row)
        for key in ("display_name", "backend_type", "source", "size_bytes", "steps"):
            setattr(row, key, r[key])
        row.captured_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(row)
        return recipe_out(row)


async def update_recipe(
    session_factory: Any, recipe_id: str, **changes: Any
) -> dict[str, Any] | None:
    """Change a recipe's context / slots / load policy / name before launching."""
    from ocabra.db.cloud import CloudModelRecipe

    async with session_factory() as session:
        row = await session.get(CloudModelRecipe, uuid.UUID(recipe_id))
        if row is None:
            return None
        steps = copy.deepcopy(row.steps or [])
        for step in steps:
            if step.get("kind") == "download":
                body = step.setdefault("body", {})
                _apply_config(body.setdefault("register_config", {}), **changes)
        row.steps = steps
        if changes.get("display_name"):
            row.display_name = changes["display_name"]
        await session.commit()
        await session.refresh(row)
        return recipe_out(row)
