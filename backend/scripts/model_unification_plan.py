"""Dry-run planner for the model/backends unification (Fase 1).

Reads production state **read-only** and prints the full mapping the migration
would apply: logical models, their deployments and default backend, what
happens to every profile / alias / preset / router target / ACL entry / agent,
how many stats rows get renamed, and every anomaly found on the way.

Nothing is written. The plan can be dumped as JSON (``--json``) so the apply
step consumes exactly what was reviewed instead of recomputing it.

Usage (inside the api container, where /data and the DB are reachable)::

    docker compose exec -T api python - < backend/scripts/model_unification_plan.py
    docker compose exec -T api python - --json < backend/scripts/model_unification_plan.py

See docs/tasks/model-backends-unification-plan.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

LLAMA_CPP_FAMILY = {"llama_cpp", "llamacpp_mks"}
LLAMA_CPP_SECTION = "llama_cpp"
OLLAMA_MODEL_LAYER = "application/vnd.ollama.image.model"
OLLAMA_PROJECTOR_LAYER = "application/vnd.ollama.image.projector"
OLLAMA_DRAFT_LAYER = "application/vnd.ollama.image.draft"
OLLAMA_REGISTRY_HOST = "registry.ollama.ai"

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

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
_WORKER_KEY_RE = re.compile(r"::[0-9a-f]{12}$")


# ── helpers ──────────────────────────────────────────────────────────────────


def split_ref(model_id: str) -> tuple[str, str]:
    backend, _, rest = model_id.partition("/")
    return backend, rest


def ollama_manifest_relpath(tag_ref: str) -> str:
    """``gemma4:26b`` → ``registry.ollama.ai/library/gemma4/26b``."""
    name, _, tag = tag_ref.partition(":")
    tag = tag or "latest"
    if "/" not in name:
        name = f"library/{name}"
    return f"{OLLAMA_REGISTRY_HOST}/{name}/{tag}"


def ollama_tag_from_relpath(relpath: str) -> str:
    parts = relpath.split("/")
    if parts and parts[0] == OLLAMA_REGISTRY_HOST:
        parts = parts[1:]
    *name_parts, tag = parts
    if name_parts and name_parts[0] == "library":
        name_parts = name_parts[1:]
    return f"{'/'.join(name_parts)}:{tag}"


def scan_ollama_manifests(root: Path) -> dict[str, dict[str, str]]:
    """Map every installed Ollama tag to its model / projector / draft blob digests."""
    out: dict[str, dict[str, str]] = {}
    manifests = root / "manifests"
    if not manifests.is_dir():
        return out
    for path in manifests.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        layers = {}
        for layer in data.get("layers") or []:
            media = layer.get("mediaType")
            digest = str(layer.get("digest") or "").replace(":", "-")
            if media == OLLAMA_MODEL_LAYER:
                layers["model"] = digest
            elif media == OLLAMA_PROJECTOR_LAYER:
                layers["projector"] = digest
            elif media == OLLAMA_DRAFT_LAYER:
                layers["draft"] = digest
        if "model" in layers:
            out[ollama_tag_from_relpath(str(path.relative_to(manifests)))] = layers
    return out


def _section(extra: dict, backend: str) -> dict:
    key = LLAMA_CPP_SECTION if backend in LLAMA_CPP_FAMILY else backend
    sec = extra.get(key)
    return sec if isinstance(sec, dict) else {}


def _opt(extra: dict, backend: str, key: str) -> Any:
    sec = _section(extra, backend)
    if key in sec:
        return sec[key]
    return extra.get(key)


def camel_to_snake(key: str) -> str:
    return _CAMEL_RE.sub("_", key).lower()


def duplicate_key_report(extra: Any, path: str = "") -> tuple[int, list[str]]:
    """Count camelCase/snake_case duplicate pairs and list the ones whose values differ."""
    if not isinstance(extra, dict):
        return 0, []
    dupes = 0
    conflicts: list[str] = []
    for key, value in extra.items():
        snake = camel_to_snake(key)
        if snake != key and snake in extra:
            dupes += 1
            if extra[snake] != value:
                conflicts.append(f"{path}{key}={value!r} vs {path}{snake}={extra[snake]!r}")
        if isinstance(value, dict):
            d, c = duplicate_key_report(value, f"{path}{key}.")
            dupes += d
            conflicts.extend(c)
    return dupes, conflicts


# ── planning ─────────────────────────────────────────────────────────────────


def describe_artifact(
    cfg: dict,
    blob_index: dict[str, list[tuple[str, str]]],
    manifests: dict[str, dict[str, str]],
    ollama_root: str,
) -> tuple[dict, tuple[str, str], list[str]]:
    """Return (artifact descriptor, grouping key, warnings) for one model_config."""
    model_id = cfg["model_id"]
    backend = cfg["backend_type"]
    _, name = split_ref(model_id)
    extra = cfg.get("extra_config") or {}
    warnings: list[str] = []

    if backend == "ollama":
        art = {"kind": "ollama_tag", "source": "ollama", "ref": name}
        layers = manifests.get(name if ":" in name else f"{name}:latest")
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
                    {"kind": "gguf", "source": "local", "path": str(model_path)},
                    ("file", str(model_path)),
                    warnings,
                )
            # ``ref`` is provisional: build_plan re-points it to the tag of the
            # Ollama deployment in the same group when there is one.
            art: dict[str, Any] = {
                "kind": "gguf",
                "source": "ollama",
                "ref": owners[0],
                "blob": digest,
            }
            if len(owners) > 1:
                art["other_tags"] = owners[1:]
            for label, value in (("mmproj", mmproj), ("draft", draft)):
                if not value:
                    continue
                sub_digest = Path(str(value)).name
                sub = blob_index.get(sub_digest, [])
                expected = "projector" if label == "mmproj" else "draft"
                if any(layer == expected and tag in owners for tag, layer in sub):
                    art[label] = {"source": "ollama", "layer": expected}
                elif any(layer == "model" for _, layer in sub):
                    # e.g. a separate Ollama model used as speculative draft
                    art[label] = {
                        "source": "ollama",
                        "ref": next(t for t, lyr in sub if lyr == "model"),
                        "layer": "model",
                    }
                else:
                    art[label] = {"source": "local", "path": str(value)}
                    warnings.append(f"{model_id}: {label} no es una capa de {owners}: {value}")
            return art, ("blob", digest), warnings
        if model_path:
            path = str(model_path)
            m = re.search(r"/huggingface/([^/]+)/([^/]+\.gguf)$", path)
            art = {"kind": "gguf", "source": "hf" if m else "local", "path": path}
            if m:
                repo_dir = m.group(1).split("--")
                art["repo"] = "/".join(repo_dir[:2])
                art["file"] = m.group(2)
            if mmproj:
                art["mmproj"] = {"source": "local", "path": str(mmproj)}
            if draft:
                art["draft"] = {"source": "local", "path": str(draft)}
            return art, ("file", path), warnings
        # No pinned path: id-based heuristics (bitnet bonsai etc.)
        return {"kind": "gguf", "source": "local", "ref": name}, ("cfg", model_id), warnings

    kind_by_backend = {
        "whisper": "ctranslate2",
        "whisper_cpp": "ggml",
        "piper": "onnx_voice",
        "diffusers": "diffusers_tree",
        "vllm": "hf_repo",
        "sglang": "hf_repo",
        "upscaler": "esrgan",
        "flashvsr": "flashvsr",
        "acestep": "acestep",
        "vibeasr": "gguf_pair",
    }
    art = {"kind": kind_by_backend.get(backend, "unknown"), "source": "local", "ref": name}
    path = _opt(extra, backend, "model_path")
    if path:
        art["path"] = str(path)
    return art, ("cfg", model_id), warnings


def build_plan(snap: dict) -> dict:
    """Pure function: snapshot of DB rows + Ollama manifests → migration plan."""
    warnings: list[str] = []
    configs = {c["model_id"]: c for c in snap["model_configs"]}
    profiles = {p["profile_id"]: p for p in snap["model_profiles"]}
    manifests: dict[str, dict[str, str]] = snap["ollama_manifests"]
    ollama_root = snap.get("ollama_root", "/data/ollama_models")

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
            warnings.append(f"grupo manual {member} → {anchor}: alguno de los dos no existe")
            continue
        if g_member is not g_anchor:
            g_anchor.extend(g_member)
            groups.remove(g_member)

    profiles_by_base: dict[str, list[dict]] = defaultdict(list)
    for p in profiles.values():
        if p.get("base_model_id"):
            profiles_by_base[p["base_model_id"]].append(p)

    # 2. per group: default backend, model_key, deployments, profile classes
    models: list[dict] = []
    deployment_map: dict[str, str] = {}  # old model_id → new deployment id
    model_of_config: dict[str, str] = {}
    profile_actions: dict[str, dict] = {}
    for group in groups:
        backends = defaultdict(list)
        for mid in group:
            backends[configs[mid]["backend_type"]].append(mid)
        for b, mids in backends.items():
            if len(mids) > 1:
                warnings.append(
                    f"grupo con dos despliegues del mismo backend {b}: {mids} — se separan"
                )
        default_backend = min(
            backends, key=lambda b: (BACKEND_PRIORITY.get(b, DEFAULT_PRIORITY), b)
        )
        default_cfg = backends[default_backend][0]

        # model_key: the default profile of the default deployment (suffix-stripped)
        own = sorted(
            profiles_by_base.get(default_cfg, []),
            key=lambda p: (not p.get("is_default"), p["profile_id"]),
        )
        if own:
            model_key = re.sub(r"-(ollama|llamacpp)$", "", own[0]["profile_id"])
        else:
            model_key = split_ref(default_cfg)[1].split("::")[0].split("/")[-1].lower()
            warnings.append(f"{default_cfg}: sin perfil; model_key derivado del id → {model_key}")

        ollama_tags = [split_ref(m)[1] for m in backends.get("ollama", [])]
        for mid in group:
            art = artifacts[mid]
            if art.get("source") == "ollama" and art.get("kind") == "gguf" and ollama_tags:
                tags = [art["ref"], *art.get("other_tags", [])]
                if ollama_tags[0] in tags:
                    art["ref"] = ollama_tags[0]
                    art["other_tags"] = [t for t in tags if t != ollama_tags[0]] or None
                    if art["other_tags"] is None:
                        del art["other_tags"]

        deployments = []
        for mid in sorted(group, key=lambda m: (configs[m]["backend_type"] != default_backend, m)):
            cfg = configs[mid]
            new_id = f"{cfg['backend_type']}/{model_key}"
            deployment_map[mid] = new_id
            model_of_config[mid] = model_key
            dupes, conflicts = duplicate_key_report(cfg.get("extra_config"))
            for c in conflicts:
                warnings.append(f"{mid}: claves camel/snake con valores DISTINTOS: {c}")
            deployments.append(
                {
                    "old_model_id": mid,
                    "new_model_id": new_id,
                    "backend": cfg["backend_type"],
                    "is_default": mid == default_cfg,
                    "artifact": artifacts[mid],
                    "preferred_gpu": cfg.get("preferred_gpu"),
                    "duplicate_case_keys": dupes,
                }
            )
            for p in profiles_by_base.get(mid, []):
                pid = p["profile_id"]
                if mid == default_cfg:
                    if p is own[0]:
                        action = {"action": "default", "new_profile_id": model_key}
                    else:
                        action = {"action": "variant", "new_profile_id": pid, "backend": None}
                elif any(p.get(f) for f in ("request_defaults", "load_overrides")):
                    # carries its own behaviour → keep it as a variant pinned to this backend
                    action = {
                        "action": "variant",
                        "new_profile_id": pid,
                        "backend": cfg["backend_type"],
                    }
                else:
                    # a plain door into a non-default deployment (the old ``-ollama`` convention)
                    action = {"action": "alias", "alias_id": pid, "backend": cfg["backend_type"]}
                action.update(model_key=model_key, old_base_model_id=mid)
                profile_actions[pid] = action

        default_row = configs[default_cfg]
        policies = {configs[m]["load_policy"] for m in group}
        if len(policies) > 1:
            warnings.append(
                f"{model_key}: load_policy distinta entre despliegues {sorted(policies)}; "
                f"se queda la del default ({default_row['load_policy']})"
            )
        models.append(
            {
                "model_key": model_key,
                "display_name": default_row.get("display_name"),
                "default_backend": default_backend,
                "load_policy": default_row["load_policy"],
                "pin_permanent": default_row.get("pin_permanent", False),
                "deployments": deployments,
            }
        )

    # 3. presets: dropped for profiles that become aliases
    dropped_presets = defaultdict(list)
    for row in snap["profile_runtime_presets"]:
        act = profile_actions.get(row["profile_id"])
        if act and act["action"] == "alias":
            dropped_presets[row["profile_id"]].append(row["preset_key"])

    # 4. existing aliases
    alias_actions = []
    for a in snap["model_profile_aliases"]:
        act = profile_actions.get(a["profile_id"])
        if act is None:
            warnings.append(f"alias {a['alias_id']} → perfil desconocido {a['profile_id']}")
            continue
        target = act.get("new_profile_id") or act["model_key"]
        alias_actions.append(
            {
                "alias_id": a["alias_id"],
                "profile_id": target,
                "backend": act.get("backend"),
                "preset_key": a.get("preset_key"),
            }
        )

    # 5. namespace collisions (profiles, aliases, model keys, routers share one namespace)
    routers = {pid for pid, p in profiles.items() if p.get("profile_kind") == "router"}
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
    for a in alias_actions:
        claim(a["alias_id"], f"alias de {a['profile_id']}")
    for r in routers:
        claim(r, f"router {r}")

    # 6. router targets
    def map_target(target: str) -> str | None:
        act = profile_actions.get(target)
        if act:
            if act["action"] == "default":
                return act["model_key"]
            if act["action"] == "variant":
                return act["new_profile_id"]
            return f"{act['backend']}/{act['model_key']}"
        if target in deployment_map:
            return deployment_map[target]
        if target in routers:
            return target
        return None

    # A target that pointed at a non-default deployment (an old ``-ollama``
    # profile) becomes the generic model, so the router follows the model's
    # default backend (user, 2026-11-07: recomendado-ha-chatbots must use the
    # default versions). Exception: when the same model is already a target of
    # that router, pinning the backend IS the point — e.g. the whisper.cpp CPU
    # fallback after the GPU whisper in recomendado-stt — so the explicit
    # deployment id is kept, in its original position.
    router_actions = []
    for rid in sorted(routers):
        old = list(profiles[rid].get("routing_targets") or [])
        generic_models = {
            profile_actions[t]["model_key"]
            for t in old
            if t in profile_actions and profile_actions[t]["action"] != "alias"
        }
        new = []
        for t in old:
            act = profile_actions.get(t)
            if act and act["action"] == "alias" and act["model_key"] not in generic_models:
                mapped = act["model_key"]
            else:
                mapped = map_target(t)
            if mapped is None:
                warnings.append(f"router {rid}: target {t!r} no existe — se elimina")
            elif mapped in new:
                warnings.append(f"router {rid}: target {t!r} duplica {mapped!r} — se elimina")
            else:
                new.append(mapped)
        router_actions.append({"router": rid, "old": old, "new": new})

    # 7. ACL
    acl_by_group: dict[str, dict] = {}
    rows_by_group = defaultdict(list)
    for row in snap["group_models"]:
        rows_by_group[row["group_name"]].append(row["model_id"])
    for gname, entries in sorted(rows_by_group.items()):
        new: set[str] = set()
        notes = []
        for e in entries:
            if e in routers:
                new.add(e)
            elif e in profile_actions:
                act = profile_actions[e]
                new.add(act["model_key"])
                if act["action"] == "variant":
                    notes.append(f"{e} (variante) → concede el modelo entero {act['model_key']}")
                elif act["action"] == "alias":
                    notes.append(
                        f"{e} (backend {act['backend']}) → concede el modelo entero {act['model_key']}"
                    )
            elif e in deployment_map:
                key = model_of_config[e]
                new.add(key)
                if deployment_map[e].split("/", 1)[0] != next(
                    m["default_backend"] for m in models if m["model_key"] == key
                ):
                    notes.append(f"{e} (despliegue no-default) → concede el modelo entero {key}")
            else:
                notes.append(f"{e}: no existe — se elimina")
        acl_by_group[gname] = {"old": sorted(entries), "new": sorted(new), "notes": notes}

    # 8. agents
    agent_actions = []
    for ag in snap["agents"]:
        agent_actions.append(
            {
                "agent": ag["slug"],
                "base_model_id": (
                    ag.get("base_model_id"),
                    model_of_config.get(ag.get("base_model_id") or ""),
                ),
                "profile_id": (
                    ag.get("profile_id"),
                    (map_target(ag["profile_id"]) if ag.get("profile_id") else None),
                ),
            }
        )

    # 9. stats: every known name (canonical id, profile, alias, model key) → model_key.
    # request_stats.model_id is historically mixed: most rows carry the public
    # name the client sent, some the canonical id. backend_type already records
    # the backend, so the deployment stays derivable after normalising.
    name_to_key: dict[str, str] = {}
    for m in models:
        name_to_key[m["model_key"]] = m["model_key"]
    for mid, key in model_of_config.items():
        name_to_key[mid] = key
    for pid, act in profile_actions.items():
        name_to_key[pid] = act["model_key"]
    for a in snap["model_profile_aliases"]:
        act = profile_actions.get(a["profile_id"])
        if act:
            name_to_key[a["alias_id"]] = act["model_key"]

    stats = {}
    for table in ("request_stats", "model_load_stats", "eviction_schedules"):
        renames: dict[str, dict] = {}
        unchanged = 0
        orphans: dict[str, int] = {}
        for mid, count in snap["stats_counts"].get(table, {}).items():
            # worker keys carry an overrides hash: ``<model_id>::<12 hex>``
            key = name_to_key.get(mid) or name_to_key.get(_WORKER_KEY_RE.sub("", mid))
            if key is None:
                if mid not in routers:
                    orphans[mid] = count
            elif key == mid:
                unchanged += count
            else:
                renames[mid] = {"to": key, "rows": count}
        stats[table] = {
            "renames": renames,
            "unchanged_rows": unchanged,
            "orphans": dict(sorted(orphans.items(), key=lambda kv: -kv[1])),
        }

    return {
        "models": sorted(models, key=lambda m: m["model_key"]),
        "profiles": profile_actions,
        "dropped_presets": dict(dropped_presets),
        "aliases": alias_actions,
        "routers": router_actions,
        "acl": acl_by_group,
        "agents": agent_actions,
        "stats": stats,
        "warnings": warnings,
    }


# ── report ───────────────────────────────────────────────────────────────────


def render(plan: dict) -> str:
    out: list[str] = []
    w = out.append
    w(f"== MODELOS ({len(plan['models'])}) ==")
    for m in plan["models"]:
        w(
            f"\n{m['model_key']}  [default: {m['default_backend']}, {m['load_policy']}{', pin' if m['pin_permanent'] else ''}]"
        )
        for d in m["deployments"]:
            star = "★" if d["is_default"] else " "
            art = d["artifact"]
            art_s = (
                art.get("kind", "?")
                + ":"
                + (
                    f"ollama:{art['ref']}"
                    if art.get("source") == "ollama"
                    else art.get("repo") or art.get("path") or art.get("ref", "")
                )
            )
            extras = "".join(f" +{k}" for k in ("mmproj", "draft") if k in art)
            w(f"  {star} {d['new_model_id']:<45} ← {d['old_model_id']}")
            w(
                f"      {art_s}{extras}"
                + (
                    f"  ({d['duplicate_case_keys']} claves camel/snake duplicadas)"
                    if d["duplicate_case_keys"]
                    else ""
                )
            )

    w("\n== PERFILES ==")
    for pid, a in sorted(plan["profiles"].items(), key=lambda kv: (kv[1]["model_key"], kv[0])):
        if a["action"] == "default":
            desc = f"default de {a['model_key']}" + (
                "" if pid == a["new_profile_id"] else f" (renombrado → {a['new_profile_id']})"
            )
        elif a["action"] == "variant":
            desc = f"variante de {a['model_key']}" + (
                f" fijada a {a['backend']}" if a["backend"] else ""
            )
        else:
            desc = f"→ ALIAS oculto de {a['model_key']} con backend {a['backend']}"
        w(f"  {pid:<55} {desc}")

    if plan["dropped_presets"]:
        w("\n== PRESETS QUE SE DESCARTAN (perfiles convertidos en alias) ==")
        for pid, keys in sorted(plan["dropped_presets"].items()):
            w(f"  {pid}: {', '.join(sorted(keys))}")

    w("\n== ALIASES EXISTENTES ==")
    for a in plan["aliases"]:
        w(
            f"  {a['alias_id']} → {a['profile_id']}"
            + (f" [{a['backend']}]" if a["backend"] else "")
            + (f" preset={a['preset_key']}" if a["preset_key"] else "")
        )

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
            w(
                f"  {ag['agent']}: base {ag['base_model_id'][0]} → {ag['base_model_id'][1]}; perfil {ag['profile_id'][0]} → {ag['profile_id'][1]}"
            )

    w("\n== ESTADÍSTICAS ==")
    for table, s in plan["stats"].items():
        rows = sum(r["rows"] for r in s["renames"].values())
        orphan_rows = sum(s["orphans"].values())
        w(
            f"  {table}: {len(s['renames'])} nombres → model_key ({rows} filas), "
            f"{s['unchanged_rows']} filas ya correctas, {orphan_rows} filas de {len(s['orphans'])} nombres desconocidos"
        )
        for name, n in list(s["orphans"].items())[:12]:
            w(f"      ? {name}: {n}")

    w(f"\n== AVISOS ({len(plan['warnings'])}) ==")
    for x in plan["warnings"]:
        w(f"  ! {x}")
    return "\n".join(out)


# ── snapshot (read-only) ─────────────────────────────────────────────────────


async def take_snapshot() -> dict:
    from sqlalchemy import text

    from ocabra.config import settings
    from ocabra.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        await session.execute(text("SET TRANSACTION READ ONLY"))

        async def rows(sql: str) -> list[dict]:
            return [dict(r._mapping) for r in (await session.execute(text(sql))).all()]

        snap = {
            "model_configs": await rows(
                "SELECT model_id, backend_type, display_name, load_policy, pin_permanent, preferred_gpu, extra_config FROM model_configs"
            ),
            "model_profiles": await rows(
                "SELECT profile_id, base_model_id, profile_kind, category, enabled, is_default, routing_targets, "
                "load_overrides, request_defaults, runtime_policy FROM model_profiles"
            ),
            "model_profile_aliases": await rows(
                "SELECT alias_id, profile_id, preset_key, hidden FROM model_profile_aliases"
            ),
            "profile_runtime_presets": await rows(
                "SELECT profile_id, preset_key FROM profile_runtime_presets"
            ),
            "group_models": await rows(
                "SELECT g.name AS group_name, gm.model_id FROM group_models gm JOIN groups g ON g.id = gm.group_id"
            ),
            "agents": await rows("SELECT slug, base_model_id, profile_id FROM agents"),
            "stats_counts": {},
        }
        for table in ("request_stats", "model_load_stats", "eviction_schedules"):
            snap["stats_counts"][table] = {
                r["model_id"]: r["n"]
                for r in await rows(
                    f"SELECT model_id, count(*) AS n FROM {table} GROUP BY model_id"
                )  # noqa: S608 — fixed table names
            }
    ollama_root = getattr(settings, "ollama_models_dir", None) or "/data/ollama_models"
    snap["ollama_root"] = ollama_root
    snap["ollama_manifests"] = scan_ollama_manifests(Path(ollama_root))
    return snap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json", action="store_true", help="print the plan as JSON instead of the report"
    )
    args = parser.parse_args()
    plan = build_plan(asyncio.run(take_snapshot()))
    if args.json:
        json.dump(plan, sys.stdout, indent=2, ensure_ascii=False, default=str)
    else:
        print(render(plan))


if __name__ == "__main__":
    main()
