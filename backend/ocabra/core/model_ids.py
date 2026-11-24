"""Model-id normalization and worker-key logic.

Pure, stateless helpers extracted from ``model_manager_helpers`` (REF-05 step 1,
closes BUG-30). ``::`` means two things in an id — the ``repo::file`` of a GGUF
deployment and the ``<id>::<hash>`` of a preset's runtime worker — so the single
place that tells them apart lives here and nowhere else.
"""

from __future__ import annotations

import hashlib
import json
import re

_DIARIZATION_SWITCHES = ("diarization_enabled", "diarizationEnabled")


def strip_diarization_overrides(load_overrides: dict | None) -> dict:
    """``load_overrides`` without the diarization switch (top level and the
    ``whisper`` section).

    Diarization is a per-request choice, not a separate worker: any whisper
    worker loads the diarizer on demand and frees it when idle. As part of the
    worker identity, ``foo`` and ``foo-diarized`` became two processes with the
    same weights; on the 3060 they did not fit together and alternating
    requests evicted one another (2026-11-11), and when one reused the other's
    process the plain profile diarized by default (2026-11-12).
    """
    out = {k: v for k, v in (load_overrides or {}).items() if k not in _DIARIZATION_SWITCHES}
    section = out.get("whisper")
    if isinstance(section, dict):
        section = {k: v for k, v in section.items() if k not in _DIARIZATION_SWITCHES}
        if section:
            out["whisper"] = section
        else:
            out.pop("whisper")
    return out


def overrides_enable_diarization(load_overrides: dict | None) -> bool:
    """Whether a profile's ``load_overrides`` turn diarization on."""
    overrides = load_overrides or {}
    section = overrides.get("whisper") if isinstance(overrides.get("whisper"), dict) else {}
    for source in (section, overrides):
        for key in _DIARIZATION_SWITCHES:
            if key in source:
                return bool(source[key])
    return False


_WORKER_SUFFIX = re.compile(r"[0-9a-f]{12}")
# Runtime suffixes older builds appended to a worker key instead of a hash.
# ``::diarize`` marked the diarizing whisper/parakeet worker until the switch
# left the worker identity (strip_diarization_overrides); request_stats still
# holds rows with it (2026-03), so a lookup by those keys must reach the base.
_LEGACY_WORKER_SUFFIXES = frozenset({"diarize"})


def is_worker_suffix(suffix: str) -> bool:
    """Whether the text after a key's last ``::`` is a runtime worker suffix
    (preset hash or legacy marker) rather than part of the deployment id."""
    return bool(_WORKER_SUFFIX.fullmatch(suffix)) or suffix in _LEGACY_WORKER_SUFFIXES


def strip_worker_suffix(worker_key: str) -> str:
    """The deployment behind a worker key: ``<id>::<hash>`` → ``<id>`` (BUG-30).

    ``::`` means two things: the 12-hex-digit hash :func:`compute_worker_key`
    appends for a preset's worker, and the ``repo::file`` of a GGUF
    deployment. Splitting at the first ``::`` turned ``repo::file::hash`` into
    ``repo`` (losing the file) and splitting at the last turned a plain
    ``repo::file`` into ``repo``. Only one trailing runtime suffix is
    stripped: a lowercase 12-hex hash, or a legacy marker (``diarize``). A GGUF
    file stem (``Model-Q4_K_M``, ``ggml-model-i2_s``) is never one of those.
    """
    base, separator, suffix = str(worker_key).rpartition("::")
    return base if separator and base and is_worker_suffix(suffix) else str(worker_key)


def is_derived_worker_key(model_id: str) -> bool:
    """Whether ``model_id`` is a runtime worker of another deployment
    (``<id>::<hash>``), not a deployment of its own (``repo::file`` included)."""
    return strip_worker_suffix(model_id) != str(model_id)


def compute_worker_key(base_model_id: str, load_overrides: dict | None) -> str:
    """Derive the stable worker key for a model profile.

    The diarization switch does not take part: see
    :func:`strip_diarization_overrides`.
    """
    load_overrides = strip_diarization_overrides(load_overrides)
    if not load_overrides:
        return base_model_id
    canonical = json.dumps(load_overrides, sort_keys=True, separators=(",", ":"))
    short_hash = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return f"{base_model_id}::{short_hash}"


def build_diarized_extra_config(base_extra_config: dict | None) -> dict:
    """Build extra_config with diarization enabled, used by profile creation."""
    merged = dict(base_extra_config or {})
    merged["diarization_enabled"] = True
    whisper_cfg = merged.get("whisper") if isinstance(merged.get("whisper"), dict) else {}
    # snake_case, and the camelCase twin removed: option readers check
    # snake_case first, so a base ``diarization_enabled: false`` would beat a
    # ``diarizationEnabled: true`` override and diarization would silently stay
    # off (extra_config is snake_case since migration 0034).
    whisper_cfg = {k: v for k, v in whisper_cfg.items() if k != "diarizationEnabled"}
    merged["whisper"] = {**whisper_cfg, "diarization_enabled": True}
    return merged
