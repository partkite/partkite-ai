"""Trigram-based fuzzy search — best for exact part numbers and SKUs."""
from __future__ import annotations

from partpilot.db import get_pool
from partpilot.models import ProductResult


async def trigram_search(
    query: str,
    limit: int = 10,
    only_in_stock: bool = False,
) -> list[ProductResult]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM trigram_search($1, $2, $3)",
            query, limit, only_in_stock,
        )
    return [_row_to_result(r, float(r["trgm_score"])) for r in rows]


def _row_to_result(r: dict, score: float) -> ProductResult:
    return ProductResult(
        id=str(r["id"]),
        product_name=r["product_name"],
        sku=r["sku"],
        price=float(r["price"]) if r["price"] is not None else None,
        source=r["source"],
        product_url=r["product_url"],
        categories=list(r["categories"]) if r["categories"] else [],
        brand=r["brand"],
        is_in_stock=r["is_in_stock"],
        raw_data=dict(r["raw_data"]) if r["raw_data"] else {},
        score=score,
    )
