"""
Async database layer — asyncpg pool configured for Supabase pgbouncer.
"""
from __future__ import annotations

import asyncpg
from asyncpg import Pool

from config import POSTGRES_DSN

_pool: Pool | None = None

# Timeout for acquiring a connection from the pool.
# Fails fast with an error rather than hanging a user request indefinitely.
_ACQUIRE_TIMEOUT = 20.0


async def init_pool() -> None:
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=POSTGRES_DSN,
        min_size=5,
        max_size=20,
        max_inactive_connection_lifetime=300,
        command_timeout=30,
        statement_cache_size=0,  # required for Supabase pgbouncer transaction mode
    )
    async with _pool.acquire() as conn:
        await conn.fetchval("SELECT 1")


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialised — call init_pool() first")
    return _pool


def acquire():
    """Acquire a pool connection with a fast-fail timeout."""
    return get_pool().acquire(timeout=_ACQUIRE_TIMEOUT)
