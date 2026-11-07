"""Model/backends unification (migration 0034): planner + applier.

Turns the legacy one-row-per-(backend, weights) inventory into logical models
(``models``) with deployments (``model_configs``, id ``<backend>/<model_key>``).

* :func:`build_plan` is a **pure** function over a snapshot of the DB and the
  Ollama manifests. It decides everything: grouping, default backend, new ids,
  artifact descriptors, normalised extra_config, profile/alias/router/ACL/stats
  rewrites, and every anomaly worth a human look (``warnings``).
* :func:`snapshot` reads that state from a **sync** SQLAlchemy connection
  (alembic's ``op.get_bind()`` or ``AsyncConnection.run_sync``).
* :func:`apply_plan` executes a plan on a sync connection, inside the caller's
  transaction. Migration 0034 calls ``apply_plan(conn, build_plan(snapshot(conn)))``.

The dry-run report is ``backend/scripts/model_unification_plan.py``.
Design and decisions: docs/tasks/model-backends-unification-plan.md.

Frozen semantics: migration 0034 depends on this module. Changing what
build_plan decides changes what that migration does on a database that has not
run it yet — treat edits here like edits to a migration.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ocabra.core.artifacts import (
    ARTIFACT_KEYS,
    KIND_BY_BACKEND,
    LLAMA_CPP_FAMILY,
    config_section_for,
    scan_ollama_manifests,
    strip_derived,
)
from ocabra.core.config_keys import camel_to_snake, normalize_case

# Lower = preferred as the default backend of a model. Ollama is kept as a
# secondary backend on purpose (decision 2026-11-07): the goal is to serve
# everything with llamacpp_mks and use Ollama as fallback + distribution.
BACKEND_PRIORITY = {"llamacpp_mks": 0, "llama_cpp": 1, "ollama": 99}
DEFAULT_PRIORITY = 10

# Deployments that serve the same model from a *different* artifact cannot be
# grouped by file identity; they are listed here explicitly.
MANUAL_GROUPS: dict[str, str] = {
    # whisper.cpp ggml q8 build of large-v3-turbo, CPU fallback of recomendado-stt
    "whisper_cpp/large-v3-turbo-q8_0-cpu": "whisper/openai/whisper-large-v3-turbo",
}

_WORKER_KEY_RE = re.compile(r"::[0-9a-f]{12}$")
_POLICY_RANK = {"on_demand": 0, "warm": 1, "pin": 2}
# Warnings that make apply_plan refuse to run.
BLOCKING_MARKERS = ("COLISIÓN", "dos despliegues del mismo backend")


# ── helpers ──────────────────────────────────────────────────────────────────


def split_ref(model_id: str) -> tuple[str, str]:
    backend, _, rest = model_id.partition("/")
    return backend, rest


def _opt(extra: dict, backend: str, key: str) -> Any:
    sec = extra.get(config_section_for(backend))
    if isinstance(sec, dict) and key in sec:
        return sec[key]
    return extra.get(key)


def duplicate_key_count(extra: Any) -> int:
    if not isinstance(extra, dict):
        return 0
    n = 0
    for key, value in extra.items():
        snake = camel_to_snake(key)
        if snake != key and snake in extra:
            n += 1
        n += duplicate_key_count(value)
    return n


def plan_extra_config(extra: Any, backend: str) -> dict | None:
    """Normalised extra_config for a deployment: snake_case, without the keys
    derived from the artifact (file paths) or moved to ``models`` (registry_metadata)."""
    if not isinstance(extra, dict):
        return None
    norm, _ = normalize_case(extra)
    norm = strip_derived(backend, norm) or {}
    norm.pop("registry_metadata", None)
    section_key = config_section_for(backend)
    section = norm.get(section_key)
    if isinstance(section, dict):
        section.pop("registry_metadata", None)
        if not section:
            norm.pop(section_key)
    return norm or None


# ── planning ─────────────────────────────────────────────────────────────────


def describe_artifact(
    cfg: dict,
    blob_index: dict[str, list[tuple[str, str]]],
    manifests: dict[str, dict[str, str]],
    ollama_root: str,
) -> tuple[dict, tuple[str, str], list[str]]:
    """Return (artifact descriptor, grouping key, warnings) for one model_config.

    ``ref`` always keeps today's backend reference (the part of the id after
    ``<backend>/``), because that is what the backend receives as model id
    (vLLM resolves the HF repo from it, piper the voice, the upscaler its
    directory, acestep its variant...). The one exception is a GGUF served
    from an Ollama blob, whose ref becomes the Ollama tag: the llama.cpp
    backends find the file through the resolved path, not through the ref.
    """
    model_id = cfg["model_id"]
    backend = cfg["backend_type"]
    _, name = split_ref(model_id)
    extra = cfg.get("extra_config") or {}
    warnings: list[str] = []

    if backend == "ollama":
        art = {"kind": "ollama_tag", "source": "ollama", "ref": name}
        layers = manifests.get(name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest")
        if layers is None:
            warnings.append(
                f"{model_id}: el tag {name} no está en los manifests de Ollama (¿borrado?)"
            )
            return art, ("ollama", name), warnings
        # Group by the GGUF blob, not by tag: one blob can be published under
        # several tags (``ollama cp``, hf.co/... pulls) and is the same file.
        return art, ("blob", layers["model"]), warnings

    if backend in LLAMA_CPP_FAMILY or backend == "bitnet":
        model_path = _opt(extra, backend, "model_path")
        mmproj = _opt(extra, backend, "mmproj_model_id") or _opt(extra, backend, "mmproj_path")
        spec = _opt(extra, backend, "speculative")
        draft = spec.get("draft_model_id") if isinstance(spec, dict) else None
        if model_path and str(model_path).startswith(f"{ollama_root}/blobs/"):
            digest = Path(str(model_path)).name
            owners = [tag for tag, layer in blob_index.get(digest, []) if layer == "model"]
            if not owners:
                warnings.append(
                    f"{model_id}: model_path apunta a un blob de Ollama que ningún manifest "
                    f"referencia como modelo ({digest[:19]}…) — ruta colgando o tag borrado"
                )
                return (
                    {"kind": "gguf", "source": "local", "ref": name, "path": str(model_path)},
                    ("file", str(model_path)),
                    warnings,
                )
            # ``ref`` is provisional: build_plan re-points it to the tag of the
            # Ollama deployment in the same group when there is one.
            art: dict[str, Any] = {"kind": "gguf", "source": "ollama", "ref": owners[0]}
            if len(owners) > 1:
                art["other_tags"] = owners[1:]
            for label, value in (("mmproj", mmproj), ("draft", draft)):
                if not value:
                    continue
                sub = blob_index.get(Path(str(value)).name, [])
                expected = "projector" if label == "mmproj" else "draft"
                if any(layer == expected and tag in owners for tag, layer in sub):
                    art[label] = {"source": "ollama", "layer": expected}
                elif any(layer == "model" for _, layer in sub):
                    # e.g. a separate Ollama model used as speculative draft
                    tag = next(t for t, lyr in sub if lyr == "model")
                    art[label] = {"source": "ollama", "ref": tag, "layer": "model"}
                else:
                    art[label] = {"source": "local", "path": str(value)}
                    warnings.append(f"{model_id}: {label} no es una capa de {owners}: {value}")
            return art, ("blob", digest), warnings
        if model_path:
            path = str(model_path)
            m = re.search(r"/huggingface/([^/]+)/([^/]+\.gguf)$", path)
            art = {"kind": "gguf", "source": "hf" if m else "local", "ref": name, "path": path}
            if m:
                art["repo"] = "/".join(m.group(1).split("--")[:2])
                art["file"] = m.group(2)
            if mmproj:
                art["mmproj"] = {"source": "local", "path": str(mmproj)}
            if draft:
                art["draft"] = {"source": "local", "path": str(draft)}
            return art, ("file", path), warnings
        # No pinned path: id-based heuristics (bitnet bonsai etc.)
        return {"kind": "gguf", "source": "local", "ref": name}, ("cfg", model_id), warnings

    art = {"kind": KIND_BY_BACKEND.get(backend, "unknown"), "source": "local", "ref": name}
    path = _opt(extra, backend, "model_path")
    if path:
        art["path"] = str(path)
    return art, ("cfg", model_id), warnings


def build_plan(snap: dict) -> dict:  # noqa: C901 — one linear pass per concern, kept together on purpose
    """Pure function: snapshot of DB rows + Ollama manifests → migration plan."""
    warnings: list[str] = []
    configs = {c["model_id"]: c for c in snap["model_configs"]}
    profiles = {p["profile_id"]: p for p in snap["model_profiles"]}
    manifests: dict[str, dict[str, str]] = snap["ollama_manifests"]
    ollama_root = snap.get("ollama_root", "/data/ollama_models")
    routers = {pid for pid, p in profiles.items() if p.get("profile_kind") == "router"}

    blob_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for tag, layers in sorted(manifests.items()):
        for layer, digest in layers.items():
            blob_index[digest].append((tag, layer))

    # 1. artifacts + grouping (union by artifact key, then manual groups)
    artifacts: dict[str, dict] = {}
    group_of_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for mid, cfg in sorted(configs.items()):
        art, key, w = describe_artifact(cfg, blob_index, manifests, ollama_root)
        artifacts[mid] = art
        group_of_key[key].append(mid)
        warnings.extend(w)
    groups: list[list[str]] = [sorted(v) for v in group_of_key.values()]
    for member, anchor in MANUAL_GROUPS.items():
        g_member = next((g for g in groups if member in g), None)
        g_anchor = next((g for g in groups if anchor in g), None)
        if g_member is None or g_anchor is None:
            continue
        if g_member is not g_anchor:
            g_anchor.extend(g_member)
            groups.remove(g_member)

    profiles_by_base: dict[str, list[dict]] = defaultdict(list)
    for p in profiles.values():
        if p.get("base_model_id"):
            profiles_by_base[p["base_model_id"]].append(p)

    # Profiles of non-default deployments that a router lists next to the
    # generic model: the backend pin is the point (CPU fallback), so they stay
    # as variants pinned to their backend. Filled in pass 2b.
    models: list[dict] = []
    deployment_map: dict[str, str] = {}  # old model_id → new deployment id
    model_of_config: dict[str, str] = {}
    default_cfg_of: dict[str, str] = {}
    profile_actions: dict[str, dict] = {}

    # 2. per group: default backend, model_key, deployments, profile classes
    for group in groups:
        backends: dict[str, list[str]] = defaultdict(list)
        for mid in group:
            backends[configs[mid]["backend_type"]].append(mid)
        for b, mids in backends.items():
            if len(mids) > 1:
                warnings.append(f"grupo con dos despliegues del mismo backend {b}: {mids}")
        default_backend = min(
            backends, key=lambda b: (BACKEND_PRIORITY.get(b, DEFAULT_PRIORITY), b)
        )
        default_cfg = backends[default_backend][0]

        own = sorted(
            profiles_by_base.get(default_cfg, []),
            key=lambda p: (not p.get("is_default"), p["profile_id"]),
        )
        if own:
            model_key = re.sub(r"-(ollama|llamacpp)$", "", own[0]["profile_id"])
        else:
            model_key = split_ref(default_cfg)[1].split("::")[0].split("/")[-1].lower()
            warnings.append(f"{default_cfg}: sin perfil; model_key derivado del id → {model_key}")
        default_cfg_of[model_key] = default_cfg

        ollama_tags = [split_ref(m)[1] for m in backends.get("ollama", [])]
        for mid in group:
            art = artifacts[mid]
            if art.get("source") == "ollama" and art.get("kind") == "gguf":
                tags = [art["ref"], *art.pop("other_tags", [])]
                if ollama_tags and ollama_tags[0] in tags:
                    art["ref"] = ollama_tags[0]

        policies = {configs[m]["load_policy"] for m in group}
        load_policy = max(policies, key=lambda pol: _POLICY_RANK.get(pol, 0))
        if len(policies) > 1:
            warnings.append(
                f"{model_key}: load_policy distinta entre despliegues {sorted(policies)}; "
                f"el default se queda con la más fuerte ({load_policy}) y el resto pasa a on_demand"
            )

        deployments = []
        for mid in sorted(group, key=lambda m: (configs[m]["backend_type"] != default_backend, m)):
            cfg = configs[mid]
            is_default = mid == default_cfg
            new_id = f"{cfg['backend_type']}/{model_key}"
            deployment_map[mid] = new_id
            model_of_config[mid] = model_key
            for c in normalize_case(cfg.get("extra_config"))[1]:
                warnings.append(f"{mid}: claves camel/snake con valores DISTINTOS, se descarta {c}")
            deployments.append(
                {
                    "old_model_id": mid,
                    "new_model_id": new_id,
                    "backend": cfg["backend_type"],
                    "is_default": is_default,
                    "artifact": artifacts[mid],
                    # Non-default deployments never preload: two warm copies of
                    # the same weights would just compete for VRAM.
                    "load_policy": load_policy if is_default else "on_demand",
                    "pin_permanent": bool(
                        is_default and any(configs[m].get("pin_permanent") for m in group)
                    ),
                    "duplicate_case_keys": duplicate_key_count(cfg.get("extra_config")),
                    "extra_config": plan_extra_config(cfg.get("extra_config"), cfg["backend_type"]),
                }
            )
            for p in profiles_by_base.get(mid, []):
                pid = p["profile_id"]
                if is_default:
                    if p is own[0]:
                        action = {"action": "default", "new_profile_id": model_key, "backend": None}
                    else:
                        action = {"action": "variant", "new_profile_id": pid, "backend": None}
                else:
                    # Profile of a non-default deployment (the old ``<name>-ollama``
                    # convention): kept as a variant pinned to its backend, presets
                    # included. They are live client entry points (176 requests in
                    # 30 days on 2026-11-07, embeddings among them, whose vectors
                    # must keep coming from the same backend), and an alias cannot
                    # force a backend until name resolution learns it (Fase 2).
                    # ``door`` marks the ones with no behaviour of their own.
                    action = {
                        "action": "variant",
                        "new_profile_id": pid,
                        "backend": cfg["backend_type"],
                        "door": not any(p.get(f) for f in ("request_defaults", "load_overrides")),
                    }
                action.update(model_key=model_key, old_base_model_id=mid, new_base_model_id=new_id)
                if p.get("load_overrides"):
                    norm, dropped = normalize_case(p["load_overrides"])
                    action["load_overrides"] = norm
                    for d in dropped:
                        warnings.append(
                            f"perfil {pid}: load_overrides camel/snake distintos, se descarta {d}"
                        )
                profile_actions[pid] = action

        default_extra = configs[default_cfg].get("extra_config") or {}
        models.append(
            {
                "model_key": model_key,
                "display_name": configs[default_cfg].get("display_name"),
                "default_backend": default_backend,
                "load_policy": load_policy,
                "registry_metadata": default_extra.get("registry_metadata"),
                "deployments": deployments,
            }
        )

    # 2b. routers. A target that pointed at a non-default deployment (an old
    # ``-ollama`` profile) becomes the generic model, so the router follows the
    # model's default backend (user, 2026-11-07: recomendado-ha-chatbots must
    # use the default versions). Exception: when the same model is already a
    # target of that router, pinning the backend IS the point (whisper.cpp CPU
    # fallback after the GPU whisper in recomendado-stt): that profile stays as
    # a variant pinned to its backend and keeps its position. (Routers only
    # accept profile ids as targets until name resolution learns deployments.)
    router_actions = []
    for rid in sorted(routers):
        old = list(profiles[rid].get("routing_targets") or [])

        def is_door(act: dict | None) -> bool:
            return bool(act) and (act["action"] == "alias" or bool(act.get("door")))

        generic_models = {
            profile_actions[t]["model_key"]
            for t in old
            if t in profile_actions and not is_door(profile_actions[t])
        }
        new: list[str] = []
        for t in old:
            act = profile_actions.get(t)
            mapped: str | None
            if act is None:
                mapped = t if t in routers else None
            elif is_door(act):
                # pinned on purpose when the generic model is also listed (CPU
                # fallback); otherwise the router follows the default backend
                mapped = t if act["model_key"] in generic_models else act["model_key"]
            elif act["action"] == "default":
                mapped = act["model_key"]
            else:
                mapped = act["new_profile_id"]
            if mapped is None:
                warnings.append(f"router {rid}: target {t!r} no existe — se elimina")
            elif mapped in new:
                warnings.append(f"router {rid}: target {t!r} duplica {mapped!r} — se elimina")
            else:
                new.append(mapped)
        router_actions.append({"router": rid, "old": old, "new": new})

    # 3. presets dropped with the profiles that become aliases
    dropped_presets: dict[str, list[str]] = defaultdict(list)
    for row in snap["profile_runtime_presets"]:
        act = profile_actions.get(row["profile_id"])
        if act and act["action"] == "alias":
            dropped_presets[row["profile_id"]].append(row["preset_key"])
    for pid, act in profile_actions.items():
        if act["action"] == "alias":
            for field in ("request_defaults", "load_overrides"):
                if profiles[pid].get(field):
                    warnings.append(f"perfil {pid} pasa a alias y PIERDE {field}")

    # 4. existing aliases
    alias_actions = []
    for a in snap["model_profile_aliases"]:
        act = profile_actions.get(a["profile_id"])
        if act is None:
            warnings.append(f"alias {a['alias_id']} → perfil desconocido {a['profile_id']}")
            continue
        if act["action"] == "alias":
            target, backend = act["model_key"], act["backend"]
        else:
            target, backend = act["new_profile_id"], act.get("backend")
        alias_actions.append(
            {
                "alias_id": a["alias_id"],
                "profile_id": target,
                "backend": backend,
                "preset_key": a.get("preset_key"),
            }
        )

    # 4b. legacy ids. Every renamed deployment id stays resolvable through a
    # hidden alias to that deployment's own profile (same backend, same
    # presets): request_stats shows clients and internal callers using
    # canonical ids (``llamacpp_mks/gemma4-26b``, ``whisper/openai/...``).
    legacy_aliases = []
    for m in models:
        for d in m["deployments"]:
            if d["old_model_id"] == d["new_model_id"]:
                continue
            own_profiles = sorted(
                (
                    (act["action"] != "default", pid, act)
                    for pid, act in profile_actions.items()
                    if act.get("new_base_model_id") == d["new_model_id"]
                    and act["action"] != "alias"
                ),
                key=lambda t: (t[0], t[1]),
            )
            target = own_profiles[0][2]["new_profile_id"] if own_profiles else m["model_key"]
            legacy_aliases.append(
                {"alias_id": d["old_model_id"], "profile_id": target, "backend": d["backend"]}
            )

    # 5. one shared namespace: model keys, profiles, aliases, routers
    names: dict[str, str] = {}

    def claim(name: str, owner: str) -> None:
        if name in names:
            warnings.append(f"COLISIÓN de nombre {name!r}: {names[name]} y {owner}")
        names[name] = owner

    for m in models:
        claim(m["model_key"], f"modelo {m['model_key']}")
    for act in profile_actions.values():
        if act["action"] == "variant":
            claim(act["new_profile_id"], f"variante de {act['model_key']}")
        elif act["action"] == "alias":
            claim(act["alias_id"], f"alias de {act['model_key']}")
    for a in alias_actions + legacy_aliases:
        claim(a["alias_id"], f"alias de {a['profile_id']}")
    for r in routers:
        claim(r, f"router {r}")

    # 6. ACL. Compatible with today's check, which matches the request's
    # profile id or its base deployment id: model ids and alias profiles map to
    # the model key (== its default profile id); variants keep their own entry
    # (granting a variant must not widen into the whole model).
    acl_by_group: dict[str, dict] = {}
    rows_by_group: dict[str, list[str]] = defaultdict(list)
    for row in snap["group_models"]:
        rows_by_group[row["group_name"]].append(row["model_id"])
    for gname, entries in sorted(rows_by_group.items()):
        new_set: set[str] = set()
        notes = []
        for e in entries:
            if e in routers:
                new_set.add(e)
            elif e in profile_actions:
                act = profile_actions[e]
                if act["action"] == "variant":
                    new_set.add(act["new_profile_id"])
                else:
                    new_set.add(act["model_key"])
                    if act["action"] == "alias":
                        notes.append(f"{e} (backend {act['backend']}) → {act['model_key']}")
            elif e in deployment_map:
                new_set.add(model_of_config[e])
            else:
                notes.append(f"{e}: no existe — se elimina")
        acl_by_group[gname] = {"old": sorted(entries), "new": sorted(new_set), "notes": notes}

    # 7. agents
    agent_actions = []
    for ag in snap["agents"]:
        pid = ag.get("profile_id")
        act = profile_actions.get(pid or "")
        new_pid = None
        if act:
            new_pid = act["model_key"] if act["action"] == "alias" else act["new_profile_id"]
            if act["action"] == "alias":
                warnings.append(
                    f"agente {ag['slug']}: su perfil {pid} pasa a alias → se apunta a {new_pid}"
                )
        agent_actions.append(
            {
                "agent": ag["slug"],
                "profile_id": [pid, new_pid if pid else None],
                "base_model_id": [
                    ag.get("base_model_id"),
                    deployment_map.get(ag.get("base_model_id") or ""),
                ],
            }
        )

    # 8. stats. request_stats.model_id → model_key (it is historically mixed:
    # mostly the public name the client sent, sometimes the canonical id or a
    # worker key); backend_type already records the backend. model_load_stats
    # and eviction_schedules are per deployment → new deployment id, keeping a
    # worker key's ``::<hash>`` suffix.
    name_to_key: dict[str, str] = {m["model_key"]: m["model_key"] for m in models}
    name_to_key.update(model_of_config)
    for pid, act in profile_actions.items():
        name_to_key[pid] = act["model_key"]
    for a in snap["model_profile_aliases"]:
        act = profile_actions.get(a["profile_id"])
        if act:
            name_to_key[a["alias_id"]] = act["model_key"]

    def stats_for(table: str, mapping: dict[str, str]) -> dict:
        renames: dict[str, dict] = {}
        unchanged = 0
        orphans: dict[str, int] = {}
        for mid, count in snap["stats_counts"].get(table, {}).items():
            base = mid if mid in mapping else _WORKER_KEY_RE.sub("", mid)
            target = mapping.get(base)
            if target is None:
                if mid not in routers:
                    orphans[mid] = count
                continue
            if table != "request_stats" and base != mid:
                target = target + mid[len(base) :]
            if target == mid:
                unchanged += count
            else:
                renames[mid] = {"to": target, "rows": count}
        return {
            "renames": renames,
            "unchanged_rows": unchanged,
            "orphans": dict(sorted(orphans.items(), key=lambda kv: -kv[1])),
        }

    stats = {
        "request_stats": stats_for("request_stats", name_to_key),
        "model_load_stats": stats_for("model_load_stats", deployment_map),
        "eviction_schedules": stats_for("eviction_schedules", deployment_map),
    }

    return {
        "models": sorted(models, key=lambda m: m["model_key"]),
        "profiles": profile_actions,
        "dropped_presets": dict(dropped_presets),
        "aliases": alias_actions,
        "legacy_aliases": legacy_aliases,
        "routers": router_actions,
        "acl": acl_by_group,
        "agents": agent_actions,
        "stats": stats,
        "warnings": warnings,
    }


# ── snapshot (read-only, sync connection) ────────────────────────────────────


def snapshot(conn: Any, ollama_root: str | None = None, scan_root: str | None = None) -> dict:
    """Read everything build_plan needs from a sync SQLAlchemy connection.

    ``ollama_root`` is the Ollama store as the api container sees it (the
    prefix stored in ``model_path``); ``scan_root`` lets a dry-run on the host
    read the same manifests from their host path.
    """
    from sqlalchemy import text

    def rows(sql: str) -> list[dict]:
        return [dict(r._mapping) for r in conn.execute(text(sql)).all()]

    snap: dict[str, Any] = {
        "model_configs": rows(
            "SELECT model_id, backend_type, display_name, load_policy, pin_permanent, "
            "preferred_gpu, extra_config FROM model_configs"
        ),
        "model_profiles": rows(
            "SELECT profile_id, base_model_id, profile_kind, is_default, routing_targets, "
            "load_overrides, request_defaults FROM model_profiles"
        ),
        "model_profile_aliases": rows(
            "SELECT alias_id, profile_id, preset_key FROM model_profile_aliases"
        ),
        "profile_runtime_presets": rows(
            "SELECT profile_id, preset_key FROM profile_runtime_presets"
        ),
        "group_models": rows(
            "SELECT g.name AS group_name, gm.model_id FROM group_models gm JOIN groups g ON g.id = gm.group_id"
        ),
        "agents": rows("SELECT slug, base_model_id, profile_id FROM agents"),
        "stats_counts": {},
    }
    for table in ("request_stats", "model_load_stats", "eviction_schedules"):
        snap["stats_counts"][table] = {
            r["model_id"]: r["n"]
            for r in rows(f"SELECT model_id, count(*) AS n FROM {table} GROUP BY model_id")  # noqa: S608 — fixed names
        }
    if ollama_root is None:
        from ocabra.config import settings

        ollama_root = getattr(settings, "ollama_shared_models_dir", "") or "/data/ollama_models"
    snap["ollama_root"] = ollama_root
    snap["ollama_manifests"] = scan_ollama_manifests(scan_root or ollama_root)
    return snap


# ── apply (sync connection, caller's transaction) ────────────────────────────


def apply_plan(conn: Any, plan: dict) -> None:  # noqa: C901 — ordered steps of one migration
    """Execute ``plan`` on ``conn``. Assumes migration 0034's schema is in place
    and that the caller owns the transaction (all-or-nothing)."""
    from sqlalchemy import text

    blocking = [w for w in plan["warnings"] if any(m in w for m in BLOCKING_MARKERS)]
    if blocking:
        raise RuntimeError("model unification plan has blocking issues:\n" + "\n".join(blocking))

    def run(sql: str, **params: Any) -> Any:
        return conn.execute(text(sql), params)

    def js(value: Any) -> str | None:
        return None if value is None else json.dumps(value)

    # Profiles about to be deleted must not take agents with them (FK cascade).
    doomed = [pid for pid, a in plan["profiles"].items() if a["action"] == "alias"]
    if doomed:
        hit = (
            run("SELECT slug FROM agents WHERE profile_id = ANY(:ids)", ids=doomed).scalars().all()
        )
        if hit:
            raise RuntimeError(f"agents use profiles that would be deleted: {hit}")

    # 1. models
    for m in plan["models"]:
        run(
            "INSERT INTO models (model_key, display_name, default_backend, registry_metadata) "
            "VALUES (:k, :d, :b, CAST(:r AS JSONB))",
            k=m["model_key"],
            d=m["display_name"],
            b=m["default_backend"],
            r=js(m["registry_metadata"]),
        )

    # 2. deployments: two-phase rename so a new id never collides with an old one
    #    still in place. FKs from model_profiles / agents follow (ON UPDATE CASCADE).
    deps = [d for m in plan["models"] for d in m["deployments"]]
    for d in deps:
        if d["old_model_id"] != d["new_model_id"]:
            run(
                "UPDATE model_configs SET model_id = :tmp WHERE model_id = :old",
                tmp=f"__mig0034__/{d['old_model_id']}",
                old=d["old_model_id"],
            )
    model_key_of = {
        d["new_model_id"]: m["model_key"] for m in plan["models"] for d in m["deployments"]
    }
    for d in deps:
        current = (
            d["old_model_id"]
            if d["old_model_id"] == d["new_model_id"]
            else f"__mig0034__/{d['old_model_id']}"
        )
        run(
            "UPDATE model_configs SET model_id = :new, model_key = :key, artifact = CAST(:art AS JSONB), "
            "extra_config = CAST(:extra AS JSONB), load_policy = :policy, pin_permanent = :pin "
            "WHERE model_id = :cur",
            new=d["new_model_id"],
            key=model_key_of[d["new_model_id"]],
            art=js(d["artifact"]),
            extra=js(d["extra_config"]),
            policy=d["load_policy"],
            pin=d["pin_permanent"],
            cur=current,
        )

    # 3. profiles
    for pid, a in plan["profiles"].items():
        if a["action"] == "alias":
            continue
        if a["new_profile_id"] != pid:
            run(
                "UPDATE model_profiles SET profile_id = :new WHERE profile_id = :old",
                new=a["new_profile_id"],
                old=pid,
            )
        run(
            "UPDATE model_profiles SET backend = :b, load_overrides = CAST(:lo AS JSONB) WHERE profile_id = :p",
            b=a.get("backend"),
            lo=js(a.get("load_overrides")),
            p=a["new_profile_id"],
        )
    # aliases that pointed at a doomed profile are re-pointed before the delete
    for a in plan["aliases"]:
        run(
            "UPDATE model_profile_aliases SET profile_id = :p, backend = :b WHERE alias_id = :a",
            p=a["profile_id"],
            b=a["backend"],
            a=a["alias_id"],
        )
    for pid, a in plan["profiles"].items():
        if a["action"] != "alias":
            continue
        run("DELETE FROM model_profiles WHERE profile_id = :p", p=pid)
        run(
            "INSERT INTO model_profile_aliases (alias_id, profile_id, preset_key, hidden, backend) "
            "VALUES (:a, :p, NULL, true, :b)",
            a=pid,
            p=a["model_key"],
            b=a["backend"],
        )

    for a in plan.get("legacy_aliases", []):
        run(
            "INSERT INTO model_profile_aliases (alias_id, profile_id, preset_key, hidden, backend) "
            "VALUES (:a, :p, NULL, true, :b)",
            a=a["alias_id"],
            p=a["profile_id"],
            b=a["backend"],
        )

    # 4. routers
    for r in plan["routers"]:
        if r["old"] != r["new"]:
            run(
                "UPDATE model_profiles SET routing_targets = CAST(:t AS JSONB) WHERE profile_id = :p",
                t=js(r["new"]),
                p=r["router"],
            )

    # 5. ACL
    for gname, acl in plan["acl"].items():
        gid = run("SELECT id FROM groups WHERE name = :n", n=gname).scalar_one()
        run("DELETE FROM group_models WHERE group_id = :g", g=gid)
        for entry in acl["new"]:
            run("INSERT INTO group_models (group_id, model_id) VALUES (:g, :m)", g=gid, m=entry)

    # 6. agents (base_model_id followed the FK cascade; profile ids only change on alias)
    for ag in plan["agents"]:
        old_pid, new_pid = ag["profile_id"]
        if old_pid and new_pid and old_pid != new_pid:
            run("UPDATE agents SET profile_id = :n WHERE slug = :s", n=new_pid, s=ag["agent"])

    # 7. stats — keep what the client asked for, then normalise
    run("UPDATE request_stats SET requested_model = model_id WHERE requested_model IS NULL")
    for table, s in plan["stats"].items():
        for old, info in s["renames"].items():
            run(f"UPDATE {table} SET model_id = :n WHERE model_id = :o", n=info["to"], o=old)  # noqa: S608


# ── report ───────────────────────────────────────────────────────────────────


def render(plan: dict) -> str:
    out: list[str] = []
    w = out.append
    w(f"== MODELOS ({len(plan['models'])}) ==")
    for m in plan["models"]:
        w(f"\n{m['model_key']}  [default: {m['default_backend']}, {m['load_policy']}]")
        for d in m["deployments"]:
            star = "★" if d["is_default"] else " "
            art = d["artifact"]
            where = (
                f"ollama:{art['ref']}"
                if art.get("source") == "ollama"
                else art.get("repo") or art.get("path") or art.get("ref", "")
            )
            extras = "".join(f" +{k}" for k in ("mmproj", "draft") if k in art)
            dupes = (
                f"  ({d['duplicate_case_keys']} claves camel/snake duplicadas)"
                if d["duplicate_case_keys"]
                else ""
            )
            w(f"  {star} {d['new_model_id']:<45} ← {d['old_model_id']}")
            w(
                f"      {art.get('kind', '?')}:{where}{extras}  ref={art.get('ref')!r}  {d['load_policy']}{dupes}"
            )

    w("\n== PERFILES ==")
    for pid, a in sorted(plan["profiles"].items(), key=lambda kv: (kv[1]["model_key"], kv[0])):
        if a["action"] == "default":
            renamed = "" if pid == a["new_profile_id"] else f" (renombrado → {a['new_profile_id']})"
            desc = f"default de {a['model_key']}{renamed}"
        elif a["action"] == "variant":
            desc = f"variante de {a['model_key']}" + (
                f" fijada a {a['backend']}" if a["backend"] else ""
            )
        else:
            desc = f"→ ALIAS oculto de {a['model_key']} (backend {a['backend']}; hasta la Fase 2 resuelve al default)"
        w(f"  {pid:<55} {desc}")

    if plan["dropped_presets"]:
        w("\n== PRESETS QUE SE DESCARTAN (perfiles convertidos en alias) ==")
        for pid, keys in sorted(plan["dropped_presets"].items()):
            w(f"  {pid}: {', '.join(sorted(keys))}")

    w("\n== ALIASES EXISTENTES ==")
    for a in plan["aliases"]:
        extra = (f" [{a['backend']}]" if a["backend"] else "") + (
            f" preset={a['preset_key']}" if a["preset_key"] else ""
        )
        w(f"  {a['alias_id']} → {a['profile_id']}{extra}")

    w(f"\n== ALIAS PARA IDS ANTIGUOS ({len(plan.get('legacy_aliases', []))}) ==")
    for a in plan.get("legacy_aliases", []):
        w(f"  {a['alias_id']} → {a['profile_id']} [{a['backend']}]")

    w("\n== ROUTERS ==")
    for r in plan["routers"]:
        mark = "" if r["old"] == r["new"] else "  (cambia)"
        w(f"  {r['router']}{mark}\n    antes: {r['old']}\n    después: {r['new']}")

    w("\n== ACL (group_models) ==")
    for g, a in plan["acl"].items():
        w(f"  {g}: {len(a['old'])} entradas → {len(a['new'])}")
        w(f"    después: {a['new']}")
        for n in a["notes"]:
            w(f"    · {n}")

    if plan["agents"]:
        w("\n== AGENTES ==")
        for ag in plan["agents"]:
            w(f"  {ag['agent']}: perfil {ag['profile_id'][0]} → {ag['profile_id'][1]}")

    w("\n== ESTADÍSTICAS ==")
    for table, s in plan["stats"].items():
        rows = sum(r["rows"] for r in s["renames"].values())
        orphan_rows = sum(s["orphans"].values())
        w(
            f"  {table}: {len(s['renames'])} nombres renombrados ({rows} filas), "
            f"{s['unchanged_rows']} filas ya correctas, {orphan_rows} filas de {len(s['orphans'])} nombres desconocidos"
        )
        for name, n in list(s["orphans"].items())[:8]:
            w(f"      ? {name}: {n}")

    w(f"\n== AVISOS ({len(plan['warnings'])}) ==")
    for x in plan["warnings"]:
        w(f"  ! {x}")
    return "\n".join(out)


__all__ = [
    "ARTIFACT_KEYS",
    "apply_plan",
    "build_plan",
    "normalize_case",
    "plan_extra_config",
    "render",
    "snapshot",
]
