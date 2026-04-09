"""Async SQLAlchemy engine and session factory."""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from server.core.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,           # set DEBUG_SQL=true env var to enable via override
    pool_pre_ping=True,   # detect stale connections
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields a database session per request.

    Route handlers are responsible for calling ``await db.commit()`` before
    returning.  This wrapper only ensures the session is closed and any
    uncommitted transaction is rolled back on unhandled exceptions.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
