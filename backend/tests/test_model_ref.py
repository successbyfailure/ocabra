import pytest

from ocabra.core.model_ref import build_model_ref, normalize_model_ref, parse_model_ref


def test_parse_model_ref_requires_canonical_prefix() -> None:
    with pytest.raises(ValueError):
        parse_model_ref("devstral-small-2:24b")


def test_normalize_model_ref_accepts_legacy_plain_name() -> None:
    canonical, backend_model = normalize_model_ref("ollama", "devstral-small-2:24b")
    assert canonical == "ollama/devstral-small-2:24b"
    assert backend_model == "devstral-small-2:24b"


def test_normalize_model_ref_accepts_repo_style_legacy_for_whisper() -> None:
    canonical, backend_model = normalize_model_ref("whisper", "openai/whisper-medium")
    assert canonical == "whisper/openai/whisper-medium"
    assert backend_model == "openai/whisper-medium"


def test_normalize_model_ref_keeps_canonical() -> None:
    canonical, backend_model = normalize_model_ref("whisper", "whisper/openai/whisper-medium")
    assert canonical == "whisper/openai/whisper-medium"
    assert backend_model == "openai/whisper-medium"


def test_build_model_ref_requires_known_backend() -> None:
    with pytest.raises(ValueError):
        build_model_ref("openai", "whisper-medium")


def test_build_model_ref_accepts_acestep_backend() -> None:
    assert build_model_ref("acestep", "turbo") == "acestep/turbo"
