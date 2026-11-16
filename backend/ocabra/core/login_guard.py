"""Brute-force protection for ``POST /ocabra/auth/login``.

Failed logins are counted in Redis per username and per client IP over a
sliding-ish window (the counter expires ``WINDOW_S`` after the first failure).
Past the limit the login answers 429 with ``Retry-After`` until the window
ends, whether the password is right or not, so guessing stops paying off.

A Redis failure lets the login through (fail open, logged): an outage of the
cache must not lock every user out of the dashboard.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

WINDOW_S = 15 * 60
MAX_FAILURES_PER_USER = 10
# Higher than per user: several people can share an office IP or a NAT.
MAX_FAILURES_PER_IP = 30


def _user_key(username: str) -> str:
    return f"auth:login_fail:user:{username.strip().lower()}"


def _ip_key(ip: str) -> str:
    return f"auth:login_fail:ip:{ip}"


async def blocked_for_seconds(username: str, ip: str | None) -> int | None:
    """Seconds until a login for ``username`` from ``ip`` is allowed again, or ``None``."""
    from ocabra.redis_client import get_redis

    checks = [(_user_key(username), MAX_FAILURES_PER_USER)]
    if ip:
        checks.append((_ip_key(ip), MAX_FAILURES_PER_IP))
    try:
        redis = get_redis()
        for key, limit in checks:
            count = await redis.get(key)
            if count is not None and int(count) >= limit:
                ttl = await redis.ttl(key)
                return ttl if ttl and ttl > 0 else WINDOW_S
    except Exception as exc:  # noqa: BLE001 - fail open, see module docstring
        logger.warning("auth_login_guard_unavailable", error=str(exc))
    return None


async def record_failure(username: str, ip: str | None) -> None:
    """Count one failed login for ``username`` and ``ip``."""
    from ocabra.redis_client import get_redis

    keys = [_user_key(username)] + ([_ip_key(ip)] if ip else [])
    try:
        redis = get_redis()
        for key in keys:
            count = await redis.incr(key)
            if count == 1:
                await redis.expire(key, WINDOW_S)
    except Exception as exc:  # noqa: BLE001 - fail open, see module docstring
        logger.warning("auth_login_guard_unavailable", error=str(exc))


async def clear_user(username: str) -> None:
    """Forget the failures of ``username`` after a successful login."""
    from ocabra.redis_client import get_redis

    try:
        await get_redis().delete(_user_key(username))
    except Exception as exc:  # noqa: BLE001 - fail open, see module docstring
        logger.warning("auth_login_guard_unavailable", error=str(exc))
