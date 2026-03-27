"""
Hybrid search: calls the DB-side hybrid_search() RRF function.
Falls back to trigram-only when no embeddings exist yet.
"""
from __future__ import annotations

from partpilot.db import get_pool
from partpilot.models import ProductResult
from partpilot.search.trigram import _row_to_result, trigram_search


async def hybrid_search(
    query: str,
    embedding: list[float],
    limit: int = 10,
    only_in_stock: bool = False,
    trgm_weight: float = 0.3,
    vec_weight: float = 0.7,
) -> list[ProductResult]:
    pool = get_pool()
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM hybrid_search($1, $2::vector, $3, $4, $5, $6)",
            query, vec_str, limit, trgm_weight, vec_weight, only_in_stock,
        )

    if not rows:
        # Fallback: no embeddings populated yet, use trigram only
        return await trigram_search(query, limit, only_in_stock)

    return [_row_to_result(r, float(r["rrf_score"])) for r in rows]
