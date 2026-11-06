"""The Makespace llama.cpp fork as its own backend type (2026-11-06).

Two separate backend_types on purpose, so both can be installed at once and
compared A/B. The risk this guards against is the *silent* one: code that asks
"is this llama.cpp?" with a bare ``== "llama_cpp"`` keeps working for upstream
and quietly skips the fork, so the fork loses adaptive admission, KV budgeting,
pool sizing and so on without any error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocabra.backends.llama_cpp_backend import LlamaCppBackend
from ocabra.backends.llamacpp_mks_backend import FORK_REF, FORK_REPO, LlamaCppMksBackend
from ocabra.core.model_ref import (
    KNOWN_BACKEND_TYPES,
    LLAMA_CPP_CONFIG_SECTION,
    LLAMA_CPP_FAMILY,
    _split_canonical_model_ref,
    is_llama_cpp_family,
)


def test_family_covers_both_and_nothing_else() -> None:
    assert LLAMA_CPP_FAMILY == {"llama_cpp", "llamacpp_mks"}
    assert is_llama_cpp_family("llama_cpp")
    assert is_llama_cpp_family("llamacpp_mks")
    # Case/whitespace tolerant: several call sites previously did their own
    # .strip().lower() before comparing, and that behaviour must survive.
    assert is_llama_cpp_family("  LLAMACPP_MKS  ")
    for other in ("bitnet", "ollama", "vllm", "", None, "llama_cpp_vulkan"):
        assert not is_llama_cpp_family(other), other
    # The fork is spelled WITHOUT the underscore after "llama" so it cannot be
    # confused with upstream's llama_cpp_<runtime> build variants. Guard the
    # spelling: "llama_cpp_mks" must not be a member, or the collision is back.
    assert not is_llama_cpp_family("llama_cpp_mks")
    assert not "llamacpp_mks".startswith("llama_cpp")


def test_backend_type_is_a_known_canonical_prefix() -> None:
    """Model ids are ``<backend_type>/<model>``; an unknown prefix makes
    _split_canonical_model_ref return None and the id stops resolving."""
    assert "llamacpp_mks" in KNOWN_BACKEND_TYPES
    assert _split_canonical_model_ref("llamacpp_mks/gemma4-26b") == (
        "llamacpp_mks",
        "gemma4-26b",
    )


def test_identity_differs_from_upstream() -> None:
    upstream, fork = LlamaCppBackend(), LlamaCppMksBackend()
    assert isinstance(fork, LlamaCppBackend), "must inherit all runtime behaviour"
    assert fork.BACKEND_TYPE == "llamacpp_mks" != upstream.BACKEND_TYPE
    # Separate install directories, or installing one would clobber the other.
    assert fork.BACKEND_DIR_NAME == "llamacpp_mks" != upstream.BACKEND_DIR_NAME


def test_install_spec_points_at_the_fork_and_reuses_the_build_recipe() -> None:
    upstream, fork = LlamaCppBackend().install_spec, LlamaCppMksBackend().install_spec
    assert fork.git_repo == FORK_REPO == "https://github.com/makespacemadrid/llama.cpp-mks"
    assert fork.git_ref == FORK_REF == "ocabra/elastic-kv"
    assert fork.git_repo != upstream.git_repo
    # The build is identical; only the source differs.
    assert fork.post_install_script == upstream.post_install_script
    assert fork.extra_bins == upstream.extra_bins
    assert fork.apt_packages == upstream.apt_packages
    assert fork.display_name != upstream.display_name



def test_fork_refuses_to_fall_back_to_the_upstream_binary(monkeypatch, tmp_path) -> None:
    """Upstream falls back to the fat-image path; the fork must not, since
    serving upstream's binary under the fork's name would advertise fork-only
    capabilities (on-demand vision) that the running server lacks."""
    monkeypatch.setattr(
        "ocabra.backends.llama_cpp_backend.settings.backends_dir", str(tmp_path / "nope")
    )
    monkeypatch.setattr(
        "ocabra.backends.llama_cpp_backend.settings.llama_cpp_server_bin",
        "/usr/local/bin/llama-server",
    )

    assert LlamaCppBackend()._resolve_server_bin() == "/usr/local/bin/llama-server"
    with pytest.raises(FileNotFoundError, match="has no installed llama-server"):
        LlamaCppMksBackend()._resolve_server_bin()


def test_fork_resolves_its_own_installed_binary(monkeypatch, tmp_path) -> None:
    backends = tmp_path / "backends"
    bin_path = backends / "llamacpp_mks" / "bin" / "llama-server"
    bin_path.parent.mkdir(parents=True)
    bin_path.write_text("#!/bin/sh\nexit 0\n")
    (backends / "llamacpp_mks" / "metadata.json").write_text(
        json.dumps({"backend_type": "llamacpp_mks", "extra_bins": {"server": str(bin_path)}})
    )
    monkeypatch.setattr("ocabra.backends.llama_cpp_backend.settings.backends_dir", str(backends))

    assert LlamaCppMksBackend()._resolve_server_bin() == str(bin_path)


def test_config_section_is_shared() -> None:
    """Both backends read the same ``extra_config`` section, so a model can be
    moved between them without rewriting its config — which is what makes the
    A/B practical. ~15 call sites depend on this being one shared key."""
    assert LLAMA_CPP_CONFIG_SECTION == "llama_cpp"


def test_registered_in_the_worker_pool() -> None:
    """A backend_type nobody registered resolves to nothing at load time."""
    main_py = Path(__file__).resolve().parents[2] / "ocabra" / "main.py"
    source = main_py.read_text(encoding="utf-8")
    assert 'register_backend("llamacpp_mks", LlamaCppMksBackend())' in source
