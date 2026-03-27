"""Vector (semantic) search via pgvector cosine similarity."""
from __future__ import annotations

from db import acquire, get_pool
from models import ProductResult
from search.trigram import _row_to_result


async def vector_search(
    embedding: list[float],
    limit: int = 10,
    only_in_stock: bool = False,
) -> list[ProductResult]:
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id, product_name, sku, price, source, product_url,
                categories, brand, is_in_stock, raw_data,
                1 - (embedding <=> $1::vector) AS vec_score
            FROM scraped_data
            WHERE embedding IS NOT NULL
              AND ($2 = false OR is_in_stock = true)
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            vec_str, only_in_stock, limit,
        )
    return [_row_to_result(r, float(r["vec_score"])) for r in rows]
