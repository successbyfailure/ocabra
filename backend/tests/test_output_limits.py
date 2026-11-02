from types import SimpleNamespace

from ocabra.core.output_limits import resolve_max_output_tokens


def test_context_length_is_not_used_as_output_limit() -> None:
    profile = SimpleNamespace(runtime_policy={"max_context_tokens": 262_144})

    assert resolve_max_output_tokens(profile, context_length=262_144) is None


def test_explicit_profile_output_limit_wins_and_is_bounded_by_context() -> None:
    profile = SimpleNamespace(runtime_policy={"max_output_tokens": 65_536})

    assert resolve_max_output_tokens(profile, context_length=262_144) == 65_536
    assert resolve_max_output_tokens(profile, context_length=32_768) == 32_768


def test_backend_limit_is_used_when_profile_has_no_override() -> None:
    profile = SimpleNamespace(runtime_policy=None)

    assert resolve_max_output_tokens(profile, backend_limit=8_192) == 8_192
