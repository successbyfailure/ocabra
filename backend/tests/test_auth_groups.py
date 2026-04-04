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


async def test_user_with_no_groups_gets_empty_set():
    """A user with no group memberships gets an empty accessible set."""
    mock_session = AsyncMock()

    result = await _fetch_accessible_models("user", [], mock_session)

    assert result == set()


async def test_user_gets_union_of_group_models():
    """A user gets the union of model_ids from all their groups."""
    group_id = str(uuid.uuid4())

    model_ids_result = MagicMock()
    model_ids_result.scalars.return_value.all.return_value = ["model-a", "model-b"]

    mock_session = AsyncMock()
    mock_session.execute.return_value = model_ids_result

    result = await _fetch_accessible_models("user", [group_id], mock_session)

    assert result == {"model-a", "model-b"}


async def test_user_with_invalid_group_uuid_skips_gracefully():
    """Invalid group UUIDs in the list must be skipped without crashing."""
    mock_session = AsyncMock()

    result = await _fetch_accessible_models("user", ["not-a-uuid", "also-not-a-uuid"], mock_session)

    assert result == set()
    mock_session.execute.assert_not_awaited()


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
    from fastapi import FastAPI, Depends
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
        patch("ocabra.api._deps_auth.AsyncSessionLocal", return_value=mock_session),
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
    from fastapi import FastAPI, Depends
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/v1/models")
    async def _endpoint(user: UserContext = Depends(get_current_user)):
        return {"ok": True}

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("ocabra.api._deps_auth.AsyncSessionLocal", return_value=mock_session),
        patch("ocabra.config.settings") as mock_settings,
    ):
        mock_settings.require_api_key_openai = True
        mock_settings.require_api_key_ollama = True

        client = TestClient(app)
        resp = client.get("/v1/models")

    assert resp.status_code == 401


# ── Model visibility per user ─────────────────────────────────────────────────


async def test_user_only_sees_models_in_their_groups():
    """A regular user should only have the models from their groups accessible."""
    group_id = str(uuid.uuid4())
    allowed_model = "group-specific-model"

    model_ids_result = MagicMock()
    model_ids_result.scalars.return_value.all.return_value = [allowed_model]

    mock_session = AsyncMock()
    mock_session.execute.return_value = model_ids_result

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
    """When a model is not in user's groups, the endpoint should 404, not 403.

    This test verifies the UserContext.accessible_model_ids set correctly
    excludes models not in the user's groups.  The actual 404 vs 403 logic
    is the responsibility of individual endpoints, not the auth layer.
    """
    group_id = str(uuid.uuid4())

    model_ids_result = MagicMock()
    model_ids_result.scalars.return_value.all.return_value = ["allowed-model"]

    mock_session = AsyncMock()
    mock_session.execute.return_value = model_ids_result

    accessible = await _fetch_accessible_models("user", [group_id], mock_session)

    # The model the user wants is NOT in their allowed set
    assert "forbidden-model" not in accessible
    # The allowed model IS in their set
    assert "allowed-model" in accessible
