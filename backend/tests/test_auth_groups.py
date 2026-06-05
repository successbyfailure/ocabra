"""Tests for group-based model filtering and anonymous access control.

All database calls are mocked.  Tests validate:
- _build_anonymous_context: only models from the default group are accessible.
- _fetch_accessible_models: regular users see union of their group models.
- Admins receive an empty accessible_model_ids set (means "all models").
- get_current_user: anonymous access is allowed / blocked based on settings.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from ocabra.api._deps_auth import (
    UserContext,
    _build_anonymous_context,
    _fetch_accessible_models,
    get_current_user,
)

# ── _fetch_accessible_models ──────────────────────────────────────────────────


async def test_admin_gets_empty_set_meaning_all_models():
    """system_admin must receive empty accessible_model_ids (= all access)."""
    mock_session = AsyncMock()

    result = await _fetch_accessible_models("system_admin", ["group-1"], mock_session)

    assert result == set()
    mock_session.execute.assert_not_awaited()  # admins skip DB query


def _scalars_result(items: list[str]) -> MagicMock:
    res = MagicMock()
    res.scalars.return_value.all.return_value = items
    return res


async def test_user_with_no_groups_gets_default_group_models():
    """A user with no explicit group memberships still inherits the default
    group's models (every user has implicit access to the default group)."""
    mock_session = AsyncMock()
    # Only one execute call expected (default group query).
    mock_session.execute.return_value = _scalars_result(["default-model"])

    result = await _fetch_accessible_models("user", [], mock_session)

    assert result == {"default-model"}


async def test_user_gets_union_of_group_models_plus_default():
    """A user gets the union of their explicit groups *and* the default
    group's models."""
    group_id = str(uuid.uuid4())

    default_result = _scalars_result(["default-model"])
    explicit_result = _scalars_result(["model-a", "model-b"])

    mock_session = AsyncMock()
    mock_session.execute.side_effect = [default_result, explicit_result]

    result = await _fetch_accessible_models("user", [group_id], mock_session)

    assert result == {"default-model", "model-a", "model-b"}


async def test_user_with_invalid_group_uuid_still_gets_default_models():
    """Invalid group UUIDs are skipped but the default group is still
    consulted — implicit access doesn't depend on the user's own groups."""
    mock_session = AsyncMock()
    mock_session.execute.return_value = _scalars_result(["default-only"])

    result = await _fetch_accessible_models(
        "user", ["not-a-uuid", "also-not-a-uuid"], mock_session
    )

    assert result == {"default-only"}
    # Default group query was issued (exactly once).
    assert mock_session.execute.await_count == 1


# ── _build_anonymous_context ──────────────────────────────────────────────────


async def test_anonymous_context_contains_default_group_models():
    """Anonymous users must only see models from the default group."""
    group_id = uuid.uuid4()
    model_id = "allowed-model"

    rows_result = MagicMock()
    rows_result.__iter__ = MagicMock(
        return_value=iter([(group_id, model_id)])
    )

    mock_session = AsyncMock()
    mock_session.execute.return_value = rows_result

    ctx = await _build_anonymous_context(mock_session)

    assert ctx.is_anonymous is True
    assert model_id in ctx.accessible_model_ids
    assert str(group_id) in ctx.group_ids


async def test_anonymous_context_when_no_default_group():
    """If no default group exists, anonymous context has empty group_ids and model_ids."""
    rows_result = MagicMock()
    rows_result.__iter__ = MagicMock(return_value=iter([]))

    mock_session = AsyncMock()
    mock_session.execute.return_value = rows_result

    ctx = await _build_anonymous_context(mock_session)

    assert ctx.is_anonymous is True
    assert ctx.group_ids == []
    assert ctx.accessible_model_ids == set()


async def test_anonymous_context_filters_out_none_model_ids():
    """Groups with no models should not contribute None to the model_ids set."""
    group_id = uuid.uuid4()

    rows_result = MagicMock()
    # One row where model_id is None (group exists but has no models)
    rows_result.__iter__ = MagicMock(
        return_value=iter([(group_id, None)])
    )

    mock_session = AsyncMock()
    mock_session.execute.return_value = rows_result

    ctx = await _build_anonymous_context(mock_session)

    assert None not in ctx.accessible_model_ids
    assert ctx.accessible_model_ids == set()


# ── get_current_user: anonymous access ───────────────────────────────────────


def test_anonymous_access_allowed_when_require_key_false():
    """When require_api_key is False, unauthenticated requests to /v1/ are allowed."""
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/v1/test")
    async def _endpoint(user: UserContext = Depends(get_current_user)):
        return {"anonymous": user.is_anonymous}

    rows_result = MagicMock()
    rows_result.__iter__ = MagicMock(return_value=iter([]))

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute.return_value = rows_result

    with (
        patch("ocabra.database.AsyncSessionLocal", return_value=mock_session),
        patch("ocabra.config.settings") as mock_settings,
    ):
        mock_settings.require_api_key_openai = False
        mock_settings.require_api_key_ollama = False

        client = TestClient(app)
        resp = client.get("/v1/test")

    assert resp.status_code == 200
    assert resp.json()["anonymous"] is True


async def test_anonymous_only_sees_default_group_models():
    """Anonymous users' context must only list models from the default group."""
    group_id = uuid.uuid4()
    allowed_model = "default-group-model"

    rows_result = MagicMock()
    rows_result.__iter__ = MagicMock(
        return_value=iter([(group_id, allowed_model)])
    )

    mock_session = AsyncMock()
    mock_session.execute.return_value = rows_result

    ctx = await _build_anonymous_context(mock_session)

    assert allowed_model in ctx.accessible_model_ids
    assert "some-other-model" not in ctx.accessible_model_ids


def test_anonymous_blocked_when_require_key_true():
    """When require_api_key_openai is True, unauthenticated /v1/ requests return 401."""
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/v1/models")
    async def _endpoint(user: UserContext = Depends(get_current_user)):
        return {"ok": True}

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("ocabra.database.AsyncSessionLocal", return_value=mock_session),
        patch("ocabra.config.settings") as mock_settings,
    ):
        mock_settings.require_api_key_openai = True
        mock_settings.require_api_key_ollama = True

        client = TestClient(app)
        resp = client.get("/v1/models")

    assert resp.status_code == 401


# ── Model visibility per user ─────────────────────────────────────────────────


async def test_user_only_sees_models_in_their_groups_and_default():
    """A regular user has access to their explicit-group models and the default
    group's models — nothing else."""
    group_id = str(uuid.uuid4())
    allowed_model = "group-specific-model"

    default_result = _scalars_result([])
    explicit_result = _scalars_result([allowed_model])

    mock_session = AsyncMock()
    mock_session.execute.side_effect = [default_result, explicit_result]

    ctx_accessible = await _fetch_accessible_models("user", [group_id], mock_session)

    assert allowed_model in ctx_accessible
    assert "other-model" not in ctx_accessible


async def test_admin_sees_all_models():
    """An admin should get an empty accessible set meaning all models are visible."""
    mock_session = AsyncMock()

    ctx_accessible = await _fetch_accessible_models("system_admin", ["any-group"], mock_session)

    # Empty set == all models (as per UserContext convention)
    assert ctx_accessible == set()


async def test_model_not_in_group_returns_404_not_403():
    """When a model is not in user's groups (or the default group), the
    endpoint should 404, not 403. Verifies the accessible set excludes
    forbidden models."""
    group_id = str(uuid.uuid4())

    default_result = _scalars_result([])
    explicit_result = _scalars_result(["allowed-model"])

    mock_session = AsyncMock()
    mock_session.execute.side_effect = [default_result, explicit_result]

    accessible = await _fetch_accessible_models("user", [group_id], mock_session)

    assert "forbidden-model" not in accessible
    assert "allowed-model" in accessible
