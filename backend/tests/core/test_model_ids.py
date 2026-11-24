"""Characterization tests for ocabra.core.model_ids (REF-05 step 1, BUG-30).

These lock the worker-suffix semantics that once broke six tests: a 12-hex
preset hash is stripped, the legacy ``::diarize`` marker is stripped, but a
GGUF ``repo::file`` deployment and a GGUF file stem must survive intact.
"""

from __future__ import annotations

import pytest

from ocabra.core import model_ids
from ocabra.core import model_manager_helpers as helpers


@pytest.mark.parametrize(
    "key,expected",
    [
        # 12-hex preset hash: stripped.
        ("gemma4:26b::0123456789ab", "gemma4:26b"),
        ("gemma4:26b::abcdef012345", "gemma4:26b"),
        # Legacy ::diarize marker: stripped.
        ("whisper-large::diarize", "whisper-large"),
        # GGUF repo::file deployment: survives (file is not a worker suffix).
        ("TheBloke/Model-GGUF::Model-Q4_K_M", "TheBloke/Model-GGUF::Model-Q4_K_M"),
        ("repo::ggml-model-i2_s", "repo::ggml-model-i2_s"),
        # repo::file::hash: only the trailing hash goes.
        ("repo::file::0123456789ab", "repo::file"),
        # GGUF file stem with no ``::``: survives untouched.
        ("Model-Q4_K_M", "Model-Q4_K_M"),
        ("ggml-model-i2_s", "ggml-model-i2_s"),
        # Non-hex / wrong-length suffixes: not worker suffixes, survive.
        ("repo::ABCDEF012345", "repo::ABCDEF012345"),  # uppercase, not [0-9a-f]
        ("repo::0123456789", "repo::0123456789"),  # 10 digits, not 12
        ("repo::0123456789abc", "repo::0123456789abc"),  # 13 digits, not 12
        ("plain", "plain"),
    ],
)
def test_strip_worker_suffix(key: str, expected: str) -> None:
    assert model_ids.strip_worker_suffix(key) == expected


def test_is_worker_suffix() -> None:
    assert model_ids.is_worker_suffix("0123456789ab") is True
    assert model_ids.is_worker_suffix("diarize") is True
    assert model_ids.is_worker_suffix("Model-Q4_K_M") is False
    assert model_ids.is_worker_suffix("0123456789") is False


def test_compute_worker_key_roundtrips_through_strip() -> None:
    # A preset over a repo::file deployment keeps the deployment after stripping.
    key = model_ids.compute_worker_key("a/b::File", {"ctx_size": 8192})
    assert key.startswith("a/b::File::")
    assert model_ids.strip_worker_suffix(key) == "a/b::File"
    # Empty / diarization-only overrides produce no suffix at all.
    assert model_ids.compute_worker_key("x", None) == "x"
    assert model_ids.compute_worker_key("x", {"diarization_enabled": True}) == "x"


def test_is_derived_worker_key() -> None:
    assert model_ids.is_derived_worker_key("gemma4:26b::0123456789ab") is True
    assert model_ids.is_derived_worker_key("gemma4:26b") is False
    assert model_ids.is_derived_worker_key("repo::Model-Q4_K_M") is False


def test_diarization_overrides() -> None:
    assert model_ids.overrides_enable_diarization({"diarization_enabled": True}) is True
    assert model_ids.overrides_enable_diarization({"whisper": {"diarizationEnabled": True}}) is True
    assert model_ids.overrides_enable_diarization({}) is False
    stripped = model_ids.strip_diarization_overrides(
        {"ctx_size": 8192, "diarization_enabled": True, "whisper": {"diarizationEnabled": False}}
    )
    assert stripped == {"ctx_size": 8192}


def test_build_diarized_extra_config() -> None:
    out = model_ids.build_diarized_extra_config({"whisper": {"diarizationEnabled": False}})
    assert out["diarization_enabled"] is True
    assert out["whisper"] == {"diarization_enabled": True}


def test_helpers_reexports_are_the_same_objects() -> None:
    # Backward-compatible re-exports: imports from model_manager_helpers must
    # resolve to the very functions now living in model_ids.
    assert helpers.strip_worker_suffix is model_ids.strip_worker_suffix
    assert helpers.compute_worker_key is model_ids.compute_worker_key
    assert helpers.is_derived_worker_key is model_ids.is_derived_worker_key
    assert helpers.build_diarized_extra_config is model_ids.build_diarized_extra_config
    assert helpers.strip_diarization_overrides is model_ids.strip_diarization_overrides
    assert helpers.overrides_enable_diarization is model_ids.overrides_enable_diarization
