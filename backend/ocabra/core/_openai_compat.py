"""Base común de los shims de compatibilidad de nombres OpenAI.

``OpenAIImageCompatRegistry`` y ``OpenAITTSCompatRegistry`` son gemelos: una
caché en memoria de filas, respaldada por PostgreSQL, que mapea nombres de
OpenAI a perfiles propios de oCabra. La única diferencia estructural es la
clave de cada entrada (el nombre de modelo solo, o el par ``(modelo, voz)``).
Esta base concentra la carga, el listado, la validación del perfil destino y
el alta/baja; cada subclase aporta su modelo ORM, su clave y su filtro de PK.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from ocabra.db.model_config import ModelProfile

logger = structlog.get_logger(__name__)

K = TypeVar("K")
R = TypeVar("R")


class OpenAICompatRegistry(Generic[K, R]):
    """Caché en memoria de las filas de un mapa de compatibilidad OpenAI.

    Las subclases definen ``_model`` (la clase ORM), ``_log_prefix`` (el prefijo
    de los eventos de log) y los hooks ``_row_key``/``_pk_filter``.
    """

    _model: type[R]
    _log_prefix: str

    def __init__(self) -> None:
        self._entries: dict[K, R] = {}

    # -- hooks por subclase ------------------------------------------------
    def _row_key(self, row: R) -> K:
        """Clave de caché de una fila (nombre de modelo o par modelo/voz)."""
        raise NotImplementedError

    def _pk_filter(self, key: K) -> list[ColumnElement[bool]]:
        """Condiciones SQL que identifican la fila de esta clave."""
        raise NotImplementedError

    # -- operaciones comunes ----------------------------------------------
    async def load_all(self, session: AsyncSession) -> None:
        rows = (await session.execute(select(self._model))).scalars().all()
        self._entries = {self._row_key(row): row for row in rows}
        logger.info(f"{self._log_prefix}_loaded", count=len(self._entries))

    def list_all(self) -> list[R]:
        return sorted(self._entries.values(), key=self._row_key)

    async def _validate_target_profile(self, session: AsyncSession, target_profile_id: str) -> None:
        row = (
            await session.execute(
                select(ModelProfile.profile_id, ModelProfile.enabled).where(
                    ModelProfile.profile_id == target_profile_id
                )
            )
        ).first()
        if row is None:
            raise ValueError(f"target_profile_id '{target_profile_id}' not found.")
        if not row.enabled:
            raise ValueError(f"target_profile_id '{target_profile_id}' is disabled.")

    async def _fetch_row(self, session: AsyncSession, key: K) -> R | None:
        return (
            await session.execute(select(self._model).where(*self._pk_filter(key)))
        ).scalar_one_or_none()

    async def _upsert_key(
        self,
        session: AsyncSession,
        key: K,
        target_profile_id: str,
        make_row: Callable[[], R],
        **log_fields: Any,
    ) -> R:
        await self._validate_target_profile(session, target_profile_id)
        # La caché refleja la BD (se puebla con load_all y se mantiene al día en
        # upsert/delete), así que basta para distinguir alta de sobrescritura
        # sin una consulta extra.
        db_entry = self._entries.get(key)
        if db_entry is None:
            db_entry = make_row()
            session.add(db_entry)
        else:
            db_entry.target_profile_id = target_profile_id  # type: ignore[attr-defined]
        await session.commit()
        await session.refresh(db_entry)
        self._entries[key] = db_entry
        logger.info(
            f"{self._log_prefix}_upserted", target_profile_id=target_profile_id, **log_fields
        )
        return db_entry

    async def _delete_key(self, session: AsyncSession, key: K, **log_fields: Any) -> bool:
        if key not in self._entries:
            return False
        db_entry = await self._fetch_row(session, key)
        if db_entry is not None:
            await session.delete(db_entry)
            await session.commit()
        del self._entries[key]
        logger.info(f"{self._log_prefix}_deleted", **log_fields)
        return True
