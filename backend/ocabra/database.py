from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from ocabra.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=settings.log_level == "DEBUG",
    pool_pre_ping=True,
    # 30 + 50 cubre con margen el pico observado del 2026-10-05 16:35 UTC
    # (281 QueuePoolTimeouts cuando estaba en 10+20): stats collector
    # middleware, downloads/*, auth revocation check, model/service
    # endpoints y el bus de eventos pueden mantener bastantes sesiones en
    # vuelo simultáneamente. pool_recycle para dar la vuelta a conexiones
    # que el servidor podría estar cerrando por idle (30 min).
    pool_size=30,
    max_overflow=50,
    pool_recycle=1800,
    pool_timeout=30.0,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass
