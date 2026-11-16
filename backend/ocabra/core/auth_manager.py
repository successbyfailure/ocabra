"""Auth manager: bcrypt hashing, JWT generation/validation, API key hashing, first-admin seed."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

import structlog

logger = structlog.get_logger(__name__)

# Lazy imports for optional heavy deps so py_compile works without them installed.
_jwt = None


def _get_jwt():
    global _jwt
    if _jwt is None:
        import jwt as _pyjwt

        _jwt = _pyjwt
    return _jwt


class AuthError(Exception):
    """Base error for authentication / authorisation failures."""


# ── Password helpers ──────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    """Hash *plain* with bcrypt (cost 12) and return the stored hash string."""
    import bcrypt

    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the bcrypt *hashed* value."""
    import bcrypt

    hashed_bytes = hashed.encode() if isinstance(hashed, str) else hashed
    return bcrypt.checkpw(plain.encode(), hashed_bytes)


async def hash_password_async(plain: str) -> str:
    """:func:`hash_password` off the event loop (bcrypt cost 12 takes ~250 ms)."""
    import asyncio

    return await asyncio.to_thread(hash_password, plain)


async def verify_password_async(plain: str, hashed: str) -> bool:
    """:func:`verify_password` off the event loop.

    On the loop, every login stalled every request in flight for ~250 ms, and
    a stream of anonymous login attempts could stall the whole API.
    """
    import asyncio

    return await asyncio.to_thread(verify_password, plain, hashed)


_DUMMY_HASH: str | None = None


async def burn_password_check(plain: str) -> None:
    """Spend the time of a real bcrypt check against a throwaway hash.

    Used when the account does not exist, so the response time does not tell
    an attacker which usernames are real.
    """
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = await hash_password_async(secrets.token_urlsafe(16))
    await verify_password_async(plain, _DUMMY_HASH)


# Passwords that ship as defaults somewhere in the project (config, compose,
# setup scripts). An account still using one is as good as open.
KNOWN_DEFAULT_PASSWORDS = frozenset({"ocabra", "change_me_in_production", "admin", "password"})
PASSWORD_MIN_LENGTH = 8


def password_problem(password: str, username: str | None = None) -> str | None:
    """Why ``password`` is not acceptable as a new password, or ``None`` if it is."""
    if len(password) < PASSWORD_MIN_LENGTH:
        return f"Password must be at least {PASSWORD_MIN_LENGTH} characters long"
    if password.lower() in KNOWN_DEFAULT_PASSWORDS:
        return "Password is a known default; choose another one"
    if username and password.strip().lower() == username.strip().lower():
        return "Password must not be the username"
    return None


# ── JWT helpers ───────────────────────────────────────────────────────────────


def create_access_token(user_id: str, role: str, remember: bool = False) -> str:
    """Create a signed JWT for *user_id* with the given *role*.

    Args:
        user_id: UUID string of the user.
        role: One of ``"user"``, ``"model_manager"``, or ``"system_admin"``.
        remember: If True, use the longer TTL (``jwt_remember_days``);
                  otherwise use ``jwt_ttl_hours``.

    Returns:
        A signed JWT string.
    """
    from ocabra.config import settings

    jwt = _get_jwt()
    now = datetime.now(UTC)
    if remember:
        exp = now + timedelta(days=settings.jwt_remember_days)
    else:
        exp = now + timedelta(hours=settings.jwt_ttl_hours)

    jti = secrets.token_hex(16)
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": exp,
        "jti": jti,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT.

    Args:
        token: The JWT string to decode.

    Returns:
        The decoded payload dictionary.

    Raises:
        AuthError: If the token is invalid, expired, or malformed.
    """
    from ocabra.config import settings

    jwt = _get_jwt()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return payload
    except Exception as exc:
        raise AuthError(f"Invalid or expired token: {exc}") from exc


# ── API key helpers ───────────────────────────────────────────────────────────


def generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key.

    Returns:
        A tuple of ``(raw_key, sha256_hash, prefix)`` where:

        - ``raw_key`` is the value to show the user once (``sk-ocabra-<24b_url_safe>``).
        - ``sha256_hash`` is what is stored in the database.
        - ``prefix`` is the first 18 characters followed by ``"…"`` for display.
    """
    raw_key = "sk-ocabra-" + secrets.token_urlsafe(24)
    key_hash = hash_api_key(raw_key)
    prefix = raw_key[:18] + "…"
    return raw_key, key_hash, prefix


def hash_api_key(key: str) -> str:
    """Return the SHA-256 hex digest of *key*."""
    return hashlib.sha256(key.encode()).hexdigest()


# ── First-admin seed ──────────────────────────────────────────────────────────


async def seed_first_admin(session) -> None:
    """Create the first admin user if the users table is empty.

    This function is idempotent: if any user already exists it does nothing.
    Credentials are read from ``settings.ocabra_admin_user`` /
    ``settings.ocabra_admin_pass``.

    Args:
        session: An active :class:`sqlalchemy.ext.asyncio.AsyncSession`.
    """
    import sqlalchemy as sa

    from ocabra.config import settings
    from ocabra.db.auth import User, UserRole

    result = await session.execute(sa.select(sa.func.count()).select_from(User))
    count = result.scalar_one()
    if count > 0:
        return

    if password_problem(settings.ocabra_admin_pass, settings.ocabra_admin_user):
        # Seeding still happens (a fresh install must be reachable), but loudly:
        # this is how the production admin ended up with the default password.
        logger.warning(
            "auth_first_admin_weak_password",
            username=settings.ocabra_admin_user,
            hint="set OCABRA_ADMIN_PASS before the first start, or change it right after logging in",
        )

    admin = User(
        username=settings.ocabra_admin_user,
        hashed_password=hash_password(settings.ocabra_admin_pass),
        role=UserRole.system_admin.value,
        is_active=True,
    )
    session.add(admin)
    await session.commit()
    logger.info(
        "auth_first_admin_created",
        username=settings.ocabra_admin_user,
    )
