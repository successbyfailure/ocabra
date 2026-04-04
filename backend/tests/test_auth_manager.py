"""Unit tests for ocabra.core.auth_manager — no database required."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocabra.core.auth_manager import (
    AuthError,
    create_access_token,
    decode_access_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    seed_first_admin,
    verify_password,
)


# ── hash_password / verify_password ──────────────────────────────────────────


def test_hash_password_is_bcrypt():
    """Hashed password must be a bcrypt hash (starts with $2b$)."""
    hashed = hash_password("secret")
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")


def test_verify_password_correct():
    """verify_password returns True for the correct plain-text password."""
    plain = "correct-horse-battery-staple"
    hashed = hash_password(plain)
    assert verify_password(plain, hashed) is True


def test_verify_password_wrong_returns_false():
    """verify_password returns False for a wrong password."""
    hashed = hash_password("correct")
    assert verify_password("wrong", hashed) is False


# ── create_access_token / decode_access_token ────────────────────────────────


def test_create_token_contains_user_id_and_role():
    """Encoded JWT payload must include sub (user_id) and role claims."""
    token = create_access_token(user_id="user-123", role="user")
    payload = decode_access_token(token)
    assert payload["sub"] == "user-123"
    assert payload["role"] == "user"


def test_decode_valid_token():
    """decode_access_token returns the full payload for a fresh token."""
    token = create_access_token(user_id="abc", role="model_manager")
    payload = decode_access_token(token)
    assert payload["sub"] == "abc"
    assert payload["role"] == "model_manager"
    assert "exp" in payload
    assert "iat" in payload
    assert "jti" in payload


def test_decode_expired_token_raises_auth_error():
    """An expired token must raise AuthError."""
    import jwt as pyjwt
    from ocabra.config import settings

    now = datetime.now(UTC)
    expired_payload = {
        "sub": "user-1",
        "role": "user",
        "iat": now - timedelta(hours=2),
        "exp": now - timedelta(hours=1),
        "jti": "expired-jti",
    }
    expired_token = pyjwt.encode(expired_payload, settings.jwt_secret, algorithm="HS256")

    with pytest.raises(AuthError):
        decode_access_token(expired_token)


def test_decode_tampered_token_raises_auth_error():
    """A token signed with a different secret must raise AuthError."""
    import jwt as pyjwt

    tampered_token = pyjwt.encode(
        {"sub": "hacker", "role": "system_admin", "exp": time.time() + 3600},
        "wrong-secret",
        algorithm="HS256",
    )

    with pytest.raises(AuthError):
        decode_access_token(tampered_token)


def test_remember_me_token_has_longer_expiry():
    """A remember=True token must expire later than a normal token."""
    from ocabra.config import settings

    token_short = create_access_token("u1", "user", remember=False)
    token_long = create_access_token("u1", "user", remember=True)

    payload_short = decode_access_token(token_short)
    payload_long = decode_access_token(token_long)

    # remember=True uses jwt_remember_days (days), normal uses jwt_ttl_hours (hours).
    assert payload_long["exp"] > payload_short["exp"]

    # The difference should be at least (jwt_remember_days - 1) days.
    min_extra_seconds = (settings.jwt_remember_days - 1) * 86400
    assert (payload_long["exp"] - payload_short["exp"]) >= min_extra_seconds


# ── generate_api_key ──────────────────────────────────────────────────────────


def test_generate_api_key_format():
    """Raw API key must start with 'sk-ocabra-'."""
    raw_key, _hash, _prefix = generate_api_key()
    assert raw_key.startswith("sk-ocabra-")


def test_generate_api_key_hash_is_sha256():
    """Stored hash must equal the SHA-256 hex digest of the raw key."""
    raw_key, key_hash, _prefix = generate_api_key()
    expected = hashlib.sha256(raw_key.encode()).hexdigest()
    assert key_hash == expected
    assert len(key_hash) == 64  # SHA-256 hex is 64 chars


def test_generate_api_key_prefix_matches_value():
    """The prefix must be the first 18 characters of the raw key followed by '…'."""
    raw_key, _hash, prefix = generate_api_key()
    assert prefix == raw_key[:18] + "…"


def test_generate_api_key_uniqueness():
    """Two successive calls must produce different raw keys and hashes."""
    raw1, hash1, prefix1 = generate_api_key()
    raw2, hash2, prefix2 = generate_api_key()
    assert raw1 != raw2
    assert hash1 != hash2


def test_hash_api_key_is_deterministic():
    """hash_api_key must return the same digest for the same input."""
    key = "sk-ocabra-testkey"
    assert hash_api_key(key) == hash_api_key(key)
    assert hash_api_key(key) == hashlib.sha256(key.encode()).hexdigest()


# ── seed_first_admin ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_seed_first_admin_creates_user():
    """seed_first_admin must create a system_admin user when the table is empty."""
    from ocabra.config import settings

    mock_session = AsyncMock()

    # Simulate empty users table (count = 0).
    count_result = MagicMock()
    count_result.scalar_one.return_value = 0
    mock_session.execute = AsyncMock(return_value=count_result)

    with patch("ocabra.core.auth_manager.hash_password", return_value="hashed"):
        await seed_first_admin(mock_session)

    mock_session.add.assert_called_once()
    mock_session.commit.assert_awaited_once()

    added_user = mock_session.add.call_args[0][0]
    assert added_user.username == settings.ocabra_admin_user
    assert added_user.hashed_password == "hashed"
    assert added_user.role == "system_admin"
    assert added_user.is_active is True


@pytest.mark.asyncio
async def test_seed_first_admin_is_idempotent():
    """seed_first_admin must do nothing when at least one user already exists."""
    mock_session = AsyncMock()

    # Simulate non-empty users table (count = 1).
    count_result = MagicMock()
    count_result.scalar_one.return_value = 1
    mock_session.execute = AsyncMock(return_value=count_result)

    await seed_first_admin(mock_session)

    mock_session.add.assert_not_called()
    mock_session.commit.assert_not_awaited()
