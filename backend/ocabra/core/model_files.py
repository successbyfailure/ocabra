"""Locate a model's files on disk (HuggingFace snapshot dirs, weight files).

Pure, stateless path-resolution helpers extracted from ``model_manager_helpers``
and ``vram_capacity`` (REF-05 step 1, DUP-01). Three places used to locate a
local HF model dir and had drifted apart; this is the single source.
"""

from __future__ import annotations

from pathlib import Path


def find_local_hf_dir(
    repo: str,
    models_dir: str | Path,
    hf_cache_dir: str | Path | None = None,
) -> Path | None:
    """Best-effort lookup of a local HuggingFace snapshot dir for ``repo``.

    Must stay in sync with ``VLLMBackend._resolve_local_model_dir`` — otherwise
    a model the backend can load fine looks "unknown" to the estimator, which
    returns 0, and the scheduler falls back to the weights-only heuristic we
    are trying to replace. Checks (same priority as the vLLM backend):

      1. ``<models_dir>/<repo>`` — direct legacy layout
      2. ``<models_dir>/huggingface/<flat>`` — flat HF layout
      3. Snapshots under ``<models_dir>/huggingface/models--<flat>/snapshots/*``
      4. ``<hf_cache_dir>/hub/models--<flat>/snapshots/<commit>`` — canonical
         HuggingFace cache layout populated by ``snapshot_download``.

    For the HF cache we prefer the commit ``refs/main`` points at (matching the
    snapshot the backend will actually load); otherwise the newest snapshot dir.
    """
    if not repo:
        return None
    # Strip the ``::<hash>`` suffix that profile worker_keys carry — the base
    # model on disk is the same regardless of load_overrides, and any lookup
    # that keeps the suffix misses the snapshot and forces the caller back to
    # the weights-only heuristic we're trying to replace.
    if "::" in repo:
        repo = repo.split("::", 1)[0]
    flat = repo.replace("/", "--")

    base = Path(models_dir)
    direct = base / repo
    if (direct / "config.json").is_file():
        return direct
    hf_base = base / "huggingface"
    for c in (hf_base / flat, hf_base / repo):
        if (c / "config.json").is_file():
            return c
    for snaps in list(hf_base.glob(f"models--{flat}/snapshots/*")) + list(
        hf_base.glob(f"*{flat}*/snapshots/*")
    ):
        if (snaps / "config.json").is_file():
            return snaps

    if hf_cache_dir:
        cache_root = Path(hf_cache_dir) / "hub" / f"models--{flat}"
        snapshots = cache_root / "snapshots"
        if snapshots.is_dir():
            commit = ""
            ref_main = cache_root / "refs" / "main"
            if ref_main.is_file():
                try:
                    commit = ref_main.read_text(encoding="utf-8").strip()
                except OSError:
                    commit = ""
            if commit:
                snap = snapshots / commit
                if (snap / "config.json").is_file():
                    return snap
            candidates = [p for p in snapshots.iterdir() if p.is_dir()]
            if candidates:
                newest = max(candidates, key=lambda p: p.stat().st_mtime)
                if (newest / "config.json").is_file():
                    return newest
    return None


def sum_hf_weight_files_mb(model_dir: Path) -> float:
    """Total MiB of the ``*.safetensors``/``*.bin`` weight files in ``model_dir``."""
    total = 0
    for pattern in ("*.safetensors", "*.bin"):
        for f in model_dir.glob(pattern):
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)
