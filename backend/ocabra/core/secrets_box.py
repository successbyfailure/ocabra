"""Encryption of the secrets oCabra stores (federation keys, MCP auth and env).

SEG-17: they used to be encrypted with a key derived from ``jwt_secret``, the
same secret that signs sessions, and that secret is persisted in the very
database whose secrets it protects (``server_config``). One leak gave both, and
a database dump alone was enough to decrypt everything.

Now the key is ``OCABRA_SECRETS_KEY``, which lives only in ``.env`` (never in the
database: see ``_ENV_ONLY_FIELDS``). The old ``jwt_secret``-derived key is still
accepted for decrypting, so nothing stored becomes unreadable, and
:func:`rotate_stored_secrets` re-encrypts everything with the new key at
startup. Without ``OCABRA_SECRETS_KEY`` it behaves as before (and says so).

Keep ``OCABRA_SECRETS_KEY`` once set: losing it means losing what it encrypted.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import structlog
from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from ocabra.config import settings

logger = structlog.get_logger(__name__)

#: (table, primary key, encrypted columns) of every stored secret.
_ENCRYPTED_COLUMNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "federation_peers",
        "id",
        ("api_key_encrypted", "local_api_key_encrypted", "join_poll_secret_encrypted"),
    ),
    ("federation_join_requests", "id", ("token_encrypted",)),
    ("mcp_servers", "id", ("env_encrypted", "auth_value_encrypted")),
    ("cloud_providers", "provider", ("api_key_encrypted",)),
)


def derive_key(secret: str) -> bytes:
    """A Fernet key from an arbitrary secret string (SHA-256, url-safe base64)."""
    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())


def fernet(legacy_secret: str | None = None) -> MultiFernet:
    """Encrypts with the dedicated key; decrypts with it or the legacy one."""
    keys = []
    if settings.ocabra_secrets_key:
        keys.append(Fernet(derive_key(settings.ocabra_secrets_key)))
    legacy = legacy_secret or settings.jwt_secret
    if legacy:
        keys.append(Fernet(derive_key(legacy)))
    return MultiFernet(keys)


def using_dedicated_key() -> bool:
    return bool(settings.ocabra_secrets_key)


async def rotate_stored_secrets(session_factory: Any) -> int:
    """Re-encrypt every stored secret with the dedicated key. Returns how many.

    A value that neither key decrypts is left untouched (and logged): it was
    already unreadable, and overwriting it would not bring it back.
    """
    if not using_dedicated_key():
        logger.warning(
            "secrets_key_not_set",
            hint="set OCABRA_SECRETS_KEY in .env: stored secrets are encrypted with the session secret",
        )
        return 0
    import sqlalchemy as sa

    box = fernet()
    primary = Fernet(derive_key(settings.ocabra_secrets_key))
    rotated = 0
    async with session_factory() as session:
        for table, pk, columns in _ENCRYPTED_COLUMNS:
            exists = (
                await session.execute(
                    sa.text("select count(*) from information_schema.tables where table_name = :t"),
                    {"t": table},
                )
            ).scalar()
            if not exists:
                continue
            rows = (
                await session.execute(sa.text(f"select {pk}, {', '.join(columns)} from {table}"))
            ).all()
            for row in rows:
                updates: dict[str, str] = {}
                for column, value in zip(columns, row[1:], strict=True):
                    if not value:
                        continue
                    token = value.encode() if isinstance(value, str) else value
                    try:
                        primary.decrypt(token)
                        continue  # already on the dedicated key
                    except InvalidToken:
                        pass
                    try:
                        updates[column] = box.rotate(token).decode()
                    except InvalidToken:
                        logger.error("stored_secret_unreadable", table=table, column=column)
                if updates:
                    sets = ", ".join(f"{column} = :{column}" for column in updates)
                    await session.execute(
                        sa.text(f"update {table} set {sets} where {pk} = :pk"),
                        {**updates, "pk": row[0]},
                    )
                    rotated += len(updates)
        await session.commit()
    if rotated:
        logger.info("stored_secrets_rotated", count=rotated)
    return rotated
